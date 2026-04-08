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
    _ocr_gray, _focus_tft, _press, _click, _d_pressed, _esc_pressed,
)
from PyQt5.QtCore import QThread, pyqtSignal

_ROOT     = Path(__file__).parent
TRAIN_DIR = _ROOT / "train"


def hashmap_path(w: int, h: int) -> Path:
    """Return the hashmap file path for a given display resolution."""
    return _ROOT / f"hashmap_{w}_{h}.json"


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

MAX_VARIANTS = 5  # max visual variants per slot+unit (1★/2★/3★, greyed out, etc.)


class HashMapper:
    """
    Thread-safe store backed by hashmap_<w>_<h>.json (one file per resolution).

    Storage format (JSON):  { "{slot}_{name}_{variant}": hash_value }  — sorted A→Z.
    Each slot+name can have up to MAX_VARIANTS=5 hashes covering visual states such as
    1-star, 2-star, 3-star, un-buyable (greyed), or other rendering variants.
    In-memory reverse index: { hash_value: unit_name }  — for O(1) roll-time lookup.

    update() rules:
      • hash already in reverse index          → no-op (variant already known).
      • score < 0.99                           → no write.
      • variants for slot+name < MAX_VARIANTS  → add as next variant, async save.
      • variants already at MAX_VARIANTS       → log and skip.
    """

    def __init__(self, path: Path):
        self._path:         Path             = path
        self._key_to_hash:  dict[str, str]   = {}   # "{slot}_{name}_{variant}" → hash, persisted
        self._hash_to_name: dict[str, str]   = {}   # hash → name, in-memory only
        self._lock = Lock()
        self.load()

    def load(self) -> int:
        """Reload from disk, rebuild reverse index. Returns entry count."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data: dict[str, str] = json.load(f)
                # Build reverse index: hash → name
                # key format: "{slot}_{name}_{variant}" e.g. "1_Jinx_0"
                reverse: dict[str, str] = {}
                for key, h in data.items():
                    after_slot = key.split("_", 1)[1] if "_" in key else key
                    name = after_slot.rsplit("_", 1)[0] if "_" in after_slot else after_slot
                    reverse[h] = name
                with self._lock:
                    self._key_to_hash  = dict(sorted(data.items()))
                    self._hash_to_name = reverse
                return len(data)
            except Exception:
                pass
        return 0

    def save(self):
        """Write key→hash map to disk (sorted). Safe to call from any thread."""
        with self._lock:
            snapshot = dict(self._key_to_hash)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def lookup(self, h: str) -> str | None:
        """O(1) hash → name lookup used during rolling."""
        with self._lock:
            return self._hash_to_name.get(h)

    def update(self, h: str, name: str, slot: int, score: float = 0.0,
               allow_overwrite: bool = True) -> tuple[str | None, bool]:
        """
        Register a hash as a visual variant of slot+name.
        Key format: "{slot}_{name}_{variant}"  e.g. "1_Jinx_0", "1_Jinx_1" …

        Write rules:
          hash already in reverse index           → no-op (variant already known).
          score < 0.99                            → no write.
          used variants < MAX_VARIANTS            → add as next free variant, async save.
          used variants == MAX_VARIANTS           → log and skip.

        allow_overwrite is kept for API compatibility but has no effect: variants are
        never overwritten, only appended.

        Returns (message, added) where:
          message – warning string if max variants reached, otherwise None
          added   – True if a new variant entry was written
        """
        if score < 0.99:
            return None, False

        prefix = f"{slot}_{name}_"
        with self._lock:
            # Already a known hash → no-op regardless of slot/name
            if h in self._hash_to_name:
                return None, False

            # Count existing variants for this slot+name
            used = {k for k in self._key_to_hash if k.startswith(prefix)}
            if len(used) >= MAX_VARIANTS:
                conflict = (
                    f"[HashMapper] MAX VARIANTS ({MAX_VARIANTS}) reached for "
                    f"'{slot}_{name}': {h[:8]} — skipping."
                )
                return conflict, False

            # Find the next free variant index (0, 1, 2, …)
            for v in range(MAX_VARIANTS):
                candidate = f"{prefix}{v}"
                if candidate not in used:
                    key = candidate
                    break

            self._key_to_hash[key] = h
            self._key_to_hash = dict(sorted(self._key_to_hash.items()))
            self._hash_to_name[h] = name

        _async_pool.submit(self.save)
        return None, True

    def remove(self, name: str, slot: int):
        """Remove all variant entries for a specific slot+name."""
        prefix = f"{slot}_{name}_"
        with self._lock:
            keys_to_remove = [k for k in self._key_to_hash if k.startswith(prefix)]
            removed_hashes: set[str] = set()
            for key in keys_to_remove:
                removed_hashes.add(self._key_to_hash.pop(key))
            # Only drop reverse-index entries not referenced by any remaining key
            for rh in removed_hashes:
                if rh not in self._key_to_hash.values():
                    self._hash_to_name.pop(rh, None)
        if keys_to_remove:
            _async_pool.submit(self.save)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._key_to_hash)

    def all_entries(self) -> list[tuple[str, str]]:
        """Returns list of (key, hash) sorted alphabetically. key = '{slot}_{name}_{variant}'."""
        with self._lock:
            return list(self._key_to_hash.items())


# Global singleton — resolution set to 1920×1080 at import time.
# Call set_active_resolution(w, h) to switch (done by the UI on settings save).
_hashmap = HashMapper(hashmap_path(1920, 1080))


def get_hashmap() -> HashMapper:
    return _hashmap


def set_active_resolution(w: int, h: int) -> None:
    """
    Switch the active hashmap to the one for resolution (w × h).
    Also updates DEFAULTS click_pos / name_regions via load_positions.
    Called by the UI whenever the user changes or saves the resolution setting.
    """
    from tft_backend import DEFAULTS, load_positions
    global _hashmap
    pos = load_positions(w, h)
    DEFAULTS["click_pos"]    = pos["click_pos"]
    DEFAULTS["name_regions"] = pos["name_regions"]
    _hashmap = HashMapper(hashmap_path(w, h))


# ── Per-slot lookup ───────────────────────────────────────────────────────────

def lookup_or_ocr(gray: np.ndarray, slot: int, threshold: float,
                  all_names: list, crop_ms: float,
                  train_mode: bool = False) -> dict:
    """
    Hash-first unit lookup with OCR fallback.

    train_mode=True:  never overwrites an existing hash entry (only adds new keys).

    Returns the same dict schema as _ocr_gray plus:
      'source':        'hash' | 'ocr'
      'hash':          32-char MD5 hex of the normalized crop
      'hash_new_entry': True when a brand-new hash key was written to the map
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
            "normalized": normalized,
            "hash_new_entry": False,
        }

    # Cache miss → fall back to OCR
    result = _ocr_gray(gray, slot, threshold, all_names, crop_ms)
    result["source"]         = "ocr"
    result["hash"]           = h
    result["hash_conflict"]  = None
    result["normalized"]     = normalized
    result["hash_new_entry"] = False
    if result["match"]:
        conflict, added = _hashmap.update(
            h, result["match"], slot, result["score"],
            allow_overwrite=not train_mode,
        )
        result["hash_new_entry"] = added
        if conflict:
            print(conflict)
            result["hash_conflict"] = conflict
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
    try:
        full_rgb = np.array(ImageGrab.grab(bbox=(band_x0, band_y0, band_x1, band_y1)))
    except Exception as e:
        print(f"[ocr_all_slots_v2] screen capture failed: {e}")
        return []
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


def save_train_sample(img_bgr: np.ndarray, result_lines: list[str],
                      results: list[dict] | None = None) -> int:
    """Save a training image (BGR numpy array) + result text + per-slot normalized
    crops. Returns sample index."""
    TRAIN_DIR.mkdir(exist_ok=True)
    idx = _next_train_idx()
    cv2.imwrite(str(TRAIN_DIR / f"{idx}_image.png"), img_bgr)
    (TRAIN_DIR / f"{idx}_result.txt").write_text(
        "\n".join(result_lines), encoding="utf-8"
    )
    if results:
        for r in results:
            norm = r.get("normalized")
            if norm is not None and norm.size > 0:
                slot = r.get("slot", 0)
                cv2.imwrite(str(TRAIN_DIR / f"{idx}_norm_{slot}.png"), norm)
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
                          (time.perf_counter() - t0) * 1000,
                          train_mode=True)
        )

    # Only save to train folder if at least one slot produced a new hash entry
    if any(r.get("hash_new_entry") for r in results):
        full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
        lines = [
            f"S{r['slot']} → {r['match'] or '?'}  source={r.get('source', '?')}"
            for r in results
        ]
        save_train_sample(full_bgr, lines, results)
        return _next_train_idx() - 1, results
    return -1, results


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
                               (time.perf_counter() - t0) * 1000,
                               train_mode=True)
        result["img_size"]      = (img_w, img_h)
        result["scaled_region"] = [x, y, w, h]
        results.append(result)

    # Only save to train folder if at least one slot produced a new hash entry
    if any(r.get("hash_new_entry") for r in results):
        lines = [
            f"S{r['slot']} → {r['match'] or '?'}  source={r.get('source', '?')}"
            f"  hash={r.get('hash', '')[:8]}"
            for r in results
        ]
        save_train_sample(img_cv, lines, results)
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
        wanted    = set(cfg.get("wanted", []))

        # ── 1. Find TFT window ────────────────────────────────
        self.status_signal.emit("Looking for TFT window…")
        if not _focus_tft():
            self.stop("⚠ TFT window not found. Is the game running?")
            self.status_signal.emit(self._reason)
            return

        # ── 2. Pre-delay ──────────────────────────────────────
        self.status_signal.emit(f"Found TFT — starting in {cfg['pre_delay']}s…")
        if self._sleep(cfg["pre_delay"]):
            self.status_signal.emit(self._reason)
            return

        # ── 3. Focus TFT once before entering the loop ────────
        if not _focus_tft():
            self.stop("⚠ TFT window not found. Is the game running?")
            self.status_signal.emit(self._reason)
            return

        while self._running:
            # ── ESC check ────────────────────────────────────
            if _esc_pressed():
                self.stop("Stopped by ESC.")
                break

            # ── 4. Scan (hash+OCR) ───────────────────────────
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
            self.shop_signal.emit(results)

            # ── 5. Buy ───────────────────────────────────────
            bought = []
            for r, pos in zip(results, cfg["click_pos"]):
                if not self._running:
                    break
                if r["match"] and r["match"] in wanted:
                    _click(pos[0], pos[1])
                    bought.append(r["match"])
                    self.found_signal.emit(
                        f"Slot {r['slot']} → {r['match']} ✓  ('{r['raw']}')")
                    if self._sleep(cfg["buy_delay"]):
                        break

            if not self._running:
                break

            hash_hits   = sum(1 for r in results if r.get("source") == "hash")
            hash_ms     = sum(r.get("crop_ms", 0) for r in results if r.get("source") == "hash")
            ocr_only_ms = sum(
                r.get("crop_ms", 0) + r.get("ocr_ms", 0)
                for r in results if r.get("source") != "hash"
            )
            bought_str = ", ".join(bought) if bought else "none"
            self.status_signal.emit(
                f"Roll {count}  "
                f"[⚡{hash_hits}/5 {hash_ms:.0f}ms | ocr {ocr_only_ms:.0f}ms]"
                f"  |  bought: {bought_str}")

            # ── 6. Roll ──────────────────────────────────────
            if auto_roll:
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

        self.status_signal.emit(self._reason)

    def stop(self, reason: str = "Stopped.") -> None:
        self._reason = reason
        self._stop_event.set()
