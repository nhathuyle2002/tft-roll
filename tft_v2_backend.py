"""
tft_v2_backend.py — Hash-first OCR-fallback unit recognition + training system.
Does NOT modify tft_backend.py (v1 code is unchanged).

  How cross-slot invariance works
  ────────────────────────────────
  The same unit name must produce the same hash regardless of which shop slot
  it appears in.  The name_regions are calibrated to the same w×h per slot,
  so raw crops are identical in size.  The normalize_crop pipeline then does:

    1. Otsu binarize      → removes background colour differences
                            (each slot has a different unit portrait behind the text)
    2. Morphological open → removes 1–2 px noise (anti-aliasing artefacts)
    3. Tight-trim to text
       bounding box       → removes the positional offset of the text within
                            the crop area, which can vary slightly per slot
    4. Resize to 128×20   → fixed canonical size, slot-independent

  After this pipeline "Jinx" in slot 1 and "Jinx" in slot 5 produce identical
  pixel arrays → identical MD5 hashes.

  Lookup strategy
  ───────────────
  1. normalize_crop → compute_hash
  2. Hit in hashmap.json → return immediately (0 Tesseract calls)
  3. Miss → run _ocr_gray (same Tesseract pipeline as v1) → if confident,
     save hash → hashmap async → return
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event
from pathlib import Path

import cv2
import numpy as np

from tft_backend import (
    TFT_UNITS, DEFAULTS, OCR_AVAILABLE,
    _ocr_gray, _focus_tft, _press, _click, _d_pressed,
)
from PyQt5.QtCore import QThread, pyqtSignal

_ROOT        = Path(__file__).parent
HASHMAP_PATH = _ROOT / "hashmap.json"
TRAIN_DIR    = _ROOT / "train"

# Shared pool for fire-and-forget async work (hashmap saves, hash updates)
_async_pool = ThreadPoolExecutor(max_workers=2)

# Canonical crop size after normalization (same for every slot / every unit)
_NORM_W, _NORM_H = 128, 20


# ── Normalization & hashing ────────────────────────────────────────────────────

def normalize_crop(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale name-crop to a canonical binary image for hashing.
    See module docstring for the cross-slot invariance explanation.
    """
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Remove isolated noise pixels (anti-aliasing artefacts between slots)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((_NORM_H, _NORM_W), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    x1  = max(0, x - pad);  y1 = max(0, y - pad)
    x2  = min(binary.shape[1], x + w + pad)
    y2  = min(binary.shape[0], y + h + pad)
    cropped = binary[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((_NORM_H, _NORM_W), dtype=np.uint8)

    # INTER_AREA for downscale, INTER_CUBIC for upscale
    interp = (cv2.INTER_AREA
               if cropped.shape[1] > _NORM_W or cropped.shape[0] > _NORM_H
               else cv2.INTER_CUBIC)
    return cv2.resize(cropped, (_NORM_W, _NORM_H), interpolation=interp)


def compute_hash(normalized: np.ndarray) -> str:
    """MD5 of the canonical binary image. Collision-resistant for TFT names."""
    return hashlib.md5(normalized.tobytes()).hexdigest()


# ── HashMapper ─────────────────────────────────────────────────────────────────

class HashMapper:
    """
    Thread-safe in-memory store backed by hashmap.json.
    Maps  hash_str → unit_name.
    All disk writes are async (daemon threads) so they never block the roll loop.
    """

    def __init__(self):
        self._map:  dict[str, str] = {}
        self._lock = Lock()
        self.load()

    def load(self) -> int:
        """Reload from disk. Returns number of entries loaded."""
        if HASHMAP_PATH.exists():
            try:
                with open(HASHMAP_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    self._map = data
                return len(data)
            except Exception:
                pass
        return 0

    def save(self):
        """Write to disk. Safe to call from any thread."""
        with self._lock:
            snapshot = dict(self._map)
        try:
            with open(HASHMAP_PATH, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def lookup(self, h: str) -> str | None:
        with self._lock:
            return self._map.get(h)

    def update(self, h: str, name: str):
        """Add entry if not already known; async disk write."""
        with self._lock:
            if h in self._map:
                return
            self._map[h] = name
        _async_pool.submit(self.save)

    def remove(self, h: str):
        with self._lock:
            self._map.pop(h, None)
        _async_pool.submit(self.save)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._map)

    def all_entries(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._map.items())


# Global singleton loaded once at import
_hashmap = HashMapper()


def get_hashmap() -> HashMapper:
    return _hashmap


# ── Per-slot lookup ───────────────────────────────────────────────────────────

def lookup_or_ocr(gray: np.ndarray, slot: int, threshold: float,
                  all_names: list, crop_ms: float) -> dict:
    """
    Hash-first unit lookup with OCR fallback.

    Returns the same dict schema as _ocr_gray plus:
      'source': 'hash' | 'ocr'
      'hash':   32-char MD5 hex of the normalized crop
    """
    normalized = normalize_crop(gray)
    h          = compute_hash(normalized)
    hit        = _hashmap.lookup(h)

    if hit is not None:
        return {
            "slot": slot, "raw": hit, "match": hit,
            "score": 1.0, "best_candidate": hit,
            "crop_ms": round(crop_ms, 1), "ocr_ms": 0.0,
            "source": "hash", "hash": h,
        }

    # Cache miss → fall back to OCR
    result = _ocr_gray(gray, slot, threshold, all_names, crop_ms)
    result["source"] = "ocr"
    result["hash"]   = h
    if result["match"]:
        _async_pool.submit(_hashmap.update, h, result["match"])
    return result


def ocr_all_slots_v2(name_regions: list, threshold: float) -> list[dict]:
    """
    Drop-in replacement for ocr_all_slots using hash-first lookup.
    Single ImageGrab; hash hits skip Tesseract entirely.
    """
    if not OCR_AVAILABLE:
        return []

    from PIL import ImageGrab

    pad       = 4
    band_x0   = min(r[0] for r in name_regions)
    band_y0   = min(r[1] for r in name_regions) - pad
    band_x1   = max(r[0] + r[2] for r in name_regions)
    band_y1   = max(r[1] + r[3] for r in name_regions) + pad
    full_rgb  = np.array(ImageGrab.grab(bbox=(band_x0, band_y0, band_x1, band_y1)))
    all_names = [n for names in TFT_UNITS.values() for n in names]

    def _slot(args):
        i, (rx, ry, rw, rh) = args
        t0   = time.perf_counter()
        y0   = ry - band_y0
        crop = full_rgb[y0 - pad : y0 + rh + pad, rx - band_x0 : rx - band_x0 + rw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return lookup_or_ocr(gray, i + 1, threshold, all_names,
                             (time.perf_counter() - t0) * 1000)

    with ThreadPoolExecutor(max_workers=len(name_regions)) as pool:
        return list(pool.map(_slot, enumerate(name_regions)))


# ── Train helpers ──────────────────────────────────────────────────────────────

def _next_train_idx() -> int:
    TRAIN_DIR.mkdir(exist_ok=True)
    nums = []
    for p in TRAIN_DIR.glob("*_image.*"):
        try:
            nums.append(int(p.stem.split("_")[0]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def save_train_sample(img_bgr: np.ndarray, result_lines: list[str]) -> int:
    """Save a training image (BGR numpy array) + result text. Returns sample index."""
    TRAIN_DIR.mkdir(exist_ok=True)
    idx = _next_train_idx()
    cv2.imwrite(str(TRAIN_DIR / f"{idx}_image.png"), img_bgr)
    (TRAIN_DIR / f"{idx}_result.txt").write_text(
        "\n".join(result_lines), encoding="utf-8"
    )
    return idx


def capture_once(name_regions: list, threshold: float) -> tuple[int, list[dict]]:
    """
    Single screen grab → hash+OCR on all slots → save train sample.
    Returns (sample_idx, results).
    """
    if not OCR_AVAILABLE:
        return -1, []

    from PIL import ImageGrab

    pad       = 4
    band_x0   = min(r[0] for r in name_regions)
    band_y0   = min(r[1] for r in name_regions) - pad
    band_x1   = max(r[0] + r[2] for r in name_regions)
    band_y1   = max(r[1] + r[3] for r in name_regions) + pad
    pil_img   = ImageGrab.grab(bbox=(band_x0, band_y0, band_x1, band_y1))
    full_rgb  = np.array(pil_img)
    all_names = [n for names in TFT_UNITS.values() for n in names]

    results = []
    for i, (rx, ry, rw, rh) in enumerate(name_regions):
        t0   = time.perf_counter()
        y0   = ry - band_y0
        crop = full_rgb[y0 - pad : y0 + rh + pad, rx - band_x0 : rx - band_x0 + rw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        results.append(
            lookup_or_ocr(gray, i + 1, threshold, all_names,
                          (time.perf_counter() - t0) * 1000)
        )

    full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"S{r['slot']} → {r['match'] or '?'}  source={r.get('source', '?')}"
        for r in results
    ]
    idx = save_train_sample(full_bgr, lines)
    return idx, results


def run_train_on_image(img_path: str, name_regions: list,
                       threshold: float) -> list[dict]:
    """
    Run hash+OCR on a file image (auto-scaled from 1920×1080).
    Saves the sample to train/. Returns list of result dicts.
    """
    if not OCR_AVAILABLE:
        return []

    from PIL import Image as _Image

    img_pil      = _Image.open(img_path).convert("RGB")
    img_w, img_h = img_pil.size
    img_cv       = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    sx, sy       = img_w / 1920, img_h / 1080
    all_names    = [n for names in TFT_UNITS.values() for n in names]
    results      = []

    for i, (rx, ry, rw, rh) in enumerate(name_regions):
        x  = int(rx * sx);  y  = int(ry * sy)
        w  = int(rw * sx);  h  = int(rh * sy)
        y0 = max(0, y);     y1 = min(img_h, y + h)
        t0 = time.perf_counter()
        crop   = img_cv[y0:y1, x:x + w]
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = lookup_or_ocr(gray, i + 1, threshold, all_names,
                               (time.perf_counter() - t0) * 1000)
        result["img_size"]      = (img_w, img_h)
        result["scaled_region"] = [x, y, w, h]
        results.append(result)

    lines = [
        f"S{r['slot']} → {r['match'] or '?'}  source={r.get('source', '?')}"
        f"  hash={r.get('hash', '')[:8]}"
        for r in results
    ]
    save_train_sample(img_cv, lines)
    return results


# ── AutoCaptureWorker ─────────────────────────────────────────────────────────

class AutoCaptureWorker(QThread):
    """Periodic screen capture → hash+OCR → save train sample → update hashmap."""
    capture_done  = pyqtSignal(int, list)   # (sample_idx, results)
    status_signal = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop_event = Event()

    def run(self) -> None:
        if not OCR_AVAILABLE:
            self.status_signal.emit("OCR not available — cannot capture.")
            return

        interval     = self.cfg.get("capture_interval", 15)
        name_regions = self.cfg.get("name_regions",  DEFAULTS["name_regions"])
        threshold    = self.cfg.get("ocr_threshold", DEFAULTS["ocr_threshold"])
        self.status_signal.emit(f"Auto-capture active (every {interval}s)")

        while True:
            # Capture immediately on start, then wait
            self.status_signal.emit("Capturing…")
            try:
                idx, results = capture_once(name_regions, threshold)
                self.capture_done.emit(idx, results)
                self.status_signal.emit(
                    f"#{idx} saved  |  hashmap: {_hashmap.size} entries"
                )
            except Exception as e:
                self.status_signal.emit(f"Capture error: {e}")
            if self._stop_event.wait(interval):
                break

    def stop(self) -> None:
        self._stop_event.set()


# ── RollWorkerV2 ──────────────────────────────────────────────────────────────
# Same logic as RollWorker but calls ocr_all_slots_v2 (hash-first).
class RollWorkerV2(QThread):
    status_signal = pyqtSignal(str)
    roll_signal   = pyqtSignal(int)
    found_signal  = pyqtSignal(str)
    shop_signal   = pyqtSignal(list)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg         = cfg
        self._stop_event = Event()
        self._reason     = "Stopped."

    @property
    def _running(self) -> bool:
        return not self._stop_event.is_set()

    def _sleep(self, seconds: float) -> bool:
        """Sleep up to `seconds` while polling ESC every 10 ms. Returns True if stopped."""
        deadline = time.perf_counter() + seconds
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return self._stop_event.is_set()
            if _esc_pressed():
                self.stop("Stopped by ESC.")
                return True
            if self._stop_event.wait(min(0.010, remaining)):
                return True

    def run(self) -> None:
        cfg       = self.cfg
        count     = 0
        auto_roll = cfg.get("auto_roll", False)
        bot_mode  = cfg.get("bot_mode", False)

        self.status_signal.emit(f"Starting in {cfg['pre_delay']}s – switch to TFT!")
        if self._sleep(cfg["pre_delay"]):
            self.status_signal.emit(self._reason)
            return

        while self._running:
            # ── ESC check at start of each loop ──────────────────
            if _esc_pressed():
                self.stop("Stopped by ESC.")
                break
            # ── Trigger ──────────────────────────────────────────
            if auto_roll:
                _focus_tft()
                _press("d")
                count += 1
                self.roll_signal.emit(count)
            else:
                self.status_signal.emit("Waiting for D key press…")
                while self._running:
                    if _d_pressed():
                        break
                    self._sleep(0.015)
                if not self._running:
                    break
                while self._running and _d_pressed():
                    self._sleep(0.015)
                if not self._running:
                    break
                count += 1
                self.roll_signal.emit(count)
                _focus_tft()

            if not self._running:
                break

            # ── Hash+OCR (BOT: parallel with shop_wait) ───────────
            t0 = time.perf_counter()
            if bot_mode:
                fut = ThreadPoolExecutor(max_workers=1).submit(
                    ocr_all_slots_v2, cfg["name_regions"], cfg["ocr_threshold"])
                self._sleep(cfg["shop_wait"])
                if not self._running:
                    break
                results = fut.result()
            else:
                if self._sleep(cfg["shop_wait"]):
                    break
                results = ocr_all_slots_v2(cfg["name_regions"], cfg["ocr_threshold"])
            ocr_ms = (time.perf_counter() - t0) * 1000
            self.shop_signal.emit(results)

            # ── Buy ──────────────────────────────────────────────
            bought = []
            for r, pos in zip(results, cfg["click_pos"]):
                if not self._running:
                    break
                if r["match"]:
                    _click(pos[0], pos[1])
                    bought.append(r["match"])
                    self.found_signal.emit(
                        f"Slot {r['slot']} → {r['match']} ✓  ('{r['raw']}')")
                    if self._sleep(cfg["buy_delay"]):
                        break

            if not self._running:
                break

            hash_hits  = sum(1 for r in results if r.get("source") == "hash")
            bought_str = ", ".join(bought) if bought else "none"
            self.status_signal.emit(
                f"Roll {count}  [{ocr_ms:.0f}ms ⚡{hash_hits}/5]  |  bought: {bought_str}")

        self.status_signal.emit(self._reason)

    def stop(self, reason: str = "Stopped.") -> None:
        self._reason = reason
        self._stop_event.set()
