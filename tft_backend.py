"""
tft_backend.py — Game data, input layer, OCR helpers, worker thread.
Imported by tft_roll_tool.py (UI) and usable standalone / in tests.
"""

import json
import time
import difflib
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _yaml = None
    _YAML_OK = False

_ROOT             = Path(__file__).parent
POSITIONS_PATH    = _ROOT / "position.yaml"
APP_SETTINGS_PATH = _ROOT / "settings.json"

# Base positions calibrated at 1920×1080 fullscreen TFT.
# ⚠️  REFERENCE CONSTANTS — do NOT change or recalculate these values.
#    They are the hand-calibrated source-of-truth from annotated screenshots.
#    All other resolutions are proportionally derived from them.
#    If recalibration is ever needed, update these constants AND the
#    1920x1080 block in position.yaml together.
_BASE_W = 1920
_BASE_H = 1080
_BASE_CLICK_POS = [
    [575, 992], [775, 992], [975, 992], [1175, 992], [1375, 992],
]
_BASE_NAME_REGIONS = [
    [480, 1045, 145, 20], [685, 1045, 145, 20], [885, 1045, 145, 20],
    [1085, 1045, 145, 20], [1290, 1045, 145, 20],
]


def _scale_positions(w: int, h: int) -> dict:
    """Proportionally scale base 1920×1080 positions to (w, h)."""
    sx, sy = w / _BASE_W, h / _BASE_H
    return {
        "click_pos": [[round(x * sx), round(y * sy)] for x, y in _BASE_CLICK_POS],
        "name_regions": [
            [round(rx * sx), round(ry * sy), max(1, round(rw * sx)), max(1, round(rh * sy))]
            for rx, ry, rw, rh in _BASE_NAME_REGIONS
        ],
    }


def load_positions(w: int, h: int) -> dict:
    """
    Return click_pos + name_regions for the given resolution.
    Reads from position.yaml first; falls back to proportional scaling.
    """
    key = f"{w}x{h}"
    if _YAML_OK and POSITIONS_PATH.exists():
        try:
            with open(POSITIONS_PATH, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            if key in data:
                res = data[key]
                click_pos    = [[res[f"slot_{i+1}"]["click_position"]["x"],
                                  res[f"slot_{i+1}"]["click_position"]["y"]] for i in range(5)]
                name_regions = [[res[f"slot_{i+1}"]["unit_name_region"]["x"],
                                  res[f"slot_{i+1}"]["unit_name_region"]["y"],
                                  res[f"slot_{i+1}"]["unit_name_region"]["w"],
                                  res[f"slot_{i+1}"]["unit_name_region"]["h"]] for i in range(5)]
                return {"click_pos": click_pos, "name_regions": name_regions}
        except Exception:
            pass
    return _scale_positions(w, h)


def save_positions(w: int, h: int, positions: dict) -> None:
    """
    Persist click_pos + name_regions for (w, h) into position.yaml.
    Existing entries for other resolutions are preserved.

    NOTE: The 1920×1080 entry is the hand-calibrated reference and is
    intentionally protected — this function will never overwrite it.
    Recalibration must be done manually in _BASE_CLICK_POS / _BASE_NAME_REGIONS
    (tft_backend.py) and in position.yaml simultaneously.
    """
    if not _YAML_OK:
        return
    key = f"{w}x{h}"
    # Guard: 1920×1080 is the canonical reference — never auto-overwrite it.
    if key == "1920x1080":
        return
    data: dict = {}
    if POSITIONS_PATH.exists():
        try:
            with open(POSITIONS_PATH, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
        except Exception:
            pass
    entry: dict = {}
    for i, ((cx, cy), (rx, ry, rw, rh)) in enumerate(
        zip(positions["click_pos"], positions["name_regions"])
    ):
        entry[f"slot_{i+1}"] = {
            "click_position":   {"x": cx, "y": cy},
            "unit_name_region": {"x": rx, "y": ry, "w": rw, "h": rh},
        }
    data[key] = entry
    with open(POSITIONS_PATH, "w", encoding="utf-8") as fh:
        _yaml.dump(data, fh, default_flow_style=None, sort_keys=True, allow_unicode=True)


def load_app_settings() -> dict:
    """Load persisted app settings (e.g. last-used resolution). Returns {} if missing."""
    if APP_SETTINGS_PATH.exists():
        try:
            with open(APP_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def save_app_settings(data: dict) -> None:
    """Persist app settings to settings.json."""
    try:
        existing: dict = {}
        if APP_SETTINGS_PATH.exists():
            with open(APP_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        existing.update(data)
        with open(APP_SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Windows-only input layer ──────────────────────────────────────────────────
# On macOS/Linux the app still launches for UI testing; input calls are no-ops.
import platform as _platform
_IS_WINDOWS = _platform.system() == "Windows"

if _IS_WINDOWS:
    import pydirectinput
    pydirectinput.FAILSAFE = False
    _user32 = ctypes.windll.user32
else:
    pydirectinput = None  # type: ignore
    _user32       = None  # type: ignore

_VK_ESCAPE = 0x1B
_VK_D      = 0x44   # 'D' virtual key code

def _d_pressed() -> bool:
    if not _IS_WINDOWS or _user32 is None:
        return False
    try:
        return bool(_user32.GetAsyncKeyState(_VK_D) & 0x8000)
    except Exception:
        return False

_TFT_TITLES   = [
    "League of Legends (TM) Client",
    "League of Legends",
    "Teamfight Tactics",
]


def _find_tft_hwnd() -> int:
    if not _IS_WINDOWS or _user32 is None:
        return 0
    for title in _TFT_TITLES:
        hwnd = _user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
    return 0


def _focus_tft() -> bool:
    """Focus the TFT window. Returns True if found, False if not running."""
    if not _IS_WINDOWS or _user32 is None:
        return True   # macOS/test — treat as success
    hwnd = _find_tft_hwnd()
    if hwnd:
        _user32.SetForegroundWindow(hwnd)
        time.sleep(0.05)
        return True
    return False


def _esc_pressed() -> bool:
    if not _IS_WINDOWS or _user32 is None:
        return False
    try:
        return bool(_user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)
    except Exception:
        return False


def _press(key: str) -> None:
    if _IS_WINDOWS and pydirectinput is not None:
        pydirectinput.press(key)


# Mouse movement: SetCursorPos (no SendInput, not intercepted by Vanguard)
# Mouse click:   legacy mouse_event API (different hook path from SendInput)
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004


def _click(x: int, y: int) -> None:
    if not _IS_WINDOWS or _user32 is None:
        return
    _user32.SetCursorPos(x, y)
    time.sleep(0.05)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


# ── Optional OCR deps ─────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import ImageGrab
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ── TFT Set 17: Space Gods – Unit roster by cost ──────────────────────────────
TFT_UNITS = {
    1: ["Aatrox", "Briar", "Caitlyn", "Cho'Gath", "Ezreal",
        "Leona", "Lissandra", "Nasus", "Poppy", "Rek'Sai",
        "Talon", "Teemo", "Twisted Fate", "Veigar"],
    2: ["Akali", "Bel'Veth", "Gnar", "Gragas", "Gwen",
        "Jax", "Jinx", "Meepsie", "Milio", "Mordekaiser",
        "Pantheon", "Pyke", "Zoe"],
    3: ["Aurora", "Diana", "Fizz", "Illaoi", "Kai'Sa",
        "Lulu", "Maokai", "Miss Fortune", "Ornn", "Rhaast",
        "Samira", "Urgot", "Viktor"],
    4: ["Aurelion Sol", "Corki", "Karma", "Kindred", "LeBlanc",
        "Master Yi", "Nami", "Nunu & Willump", "Rammus", "Riven",
        "Tahm Kench", "The Mighty Mech", "Xayah"],
    5: ["Bard", "Blitzcrank", "Fiora", "Graves", "Jhin",
        "Morgana", "Shen", "Sona", "Vex", "Zed"],
}

COST_COLOR = {1: "#aaaaaa", 2: "#56d364", 3: "#58a6ff", 4: "#bc8cff", 5: "#ffd700"}
COST_BG    = {1: "#2d2d2d", 2: "#12291a", 3: "#0d1f33", 4: "#1e1033", 5: "#2e2500"}
COST_LABEL = {1: "● Cost 1", 2: "●● Cost 2", 3: "●●● Cost 3",
              4: "◆◆◆◆ Cost 4", 5: "◆◆◆◆◆ Cost 5"}


# ── Runtime defaults ──────────────────────────────────────────────────────────
# Timing values are static; click_pos / name_regions are overwritten at startup
# by load_positions() for the active resolution (default 1920×1080).
DEFAULTS = {
    # Timing  (Human mode defaults)
    "roll_delay":    1.5,
    "pre_delay":     1,
    "shop_wait":     0.15,
    "buy_delay":     0.05,

    # OCR
    "ocr_threshold": 0.50,

    # Positions — initialised below from position.yaml (1920×1080 as fallback)
    "click_pos":    [[575, 992], [775, 992], [975, 992], [1175, 992], [1375, 992]],
    "name_regions": [[480, 1045, 145, 20], [685, 1045, 145, 20],
                     [885, 1045, 145, 20], [1085, 1045, 145, 20],
                     [1290, 1045, 145, 20]],
}

# Populate positions from yaml immediately so importers get the right values.
_pos_init = load_positions(1920, 1080)
DEFAULTS["click_pos"]    = _pos_init["click_pos"]
DEFAULTS["name_regions"] = _pos_init["name_regions"]


# ── OCR helpers ───────────────────────────────────────────────────────────────
_OCR_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' &-"
_OCR_CONFIG = f'--psm 7 --oem 1 -c "tessedit_char_whitelist={_OCR_WL}"'


def _preprocess_variants(gray: "np.ndarray") -> "list[np.ndarray]":
    """Return binarised variants of a grayscale crop for OCR."""
    results = []
    for scale in (3, 4):
        scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
        _, v1 = cv2.threshold(scaled, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, v2 = cv2.threshold(scaled, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v3 = cv2.adaptiveThreshold(scaled, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
        results.extend([v1, v2, v3])
    return results


def _best_fuzzy(text: str, candidates: list[str]) -> tuple[str, float]:
    """Return (best_name, score) from candidates for the given text."""
    best_name, best_score = "", 0.0
    low = text.lower()
    for name in candidates:
        s = difflib.SequenceMatcher(None, low, name.lower()).ratio()
        if s > best_score:
            best_score, best_name = s, name
    return best_name, best_score


def ocr_unit_name(region: list, threshold: float) -> str:
    """
    Screenshot one name region and return the best raw OCR string.
    Uses fuzzy-match against the full unit roster to pick the best variant.
    """
    if not OCR_AVAILABLE:
        return ""
    x, y, w, h = region
    pad  = 4
    img  = np.array(ImageGrab.grab(bbox=(x, y - pad, x + w, y + h + pad)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    all_names    = [n for names in TFT_UNITS.values() for n in names]
    best_raw     = ""
    best_score   = 0.0

    for variant in _preprocess_variants(gray):
        raw = pytesseract.image_to_string(variant, config=_OCR_CONFIG).strip()
        if not raw:
            continue
        _, score = _best_fuzzy(raw, all_names)
        if score > best_score:
            best_score = score
            best_raw   = raw

    return best_raw


def _ocr_gray(gray, slot: int, threshold: float, all_names: list,
              crop_ms: float) -> dict:
    """
    Core per-slot OCR: run Tesseract on a grayscale crop and return a result dict.
    Used by both ocr_all_slots (live screen) and ocr_from_image_file (file).
    """
    t_ocr = time.perf_counter()
    best_raw, best_score = "", 0.0
    for variant in _preprocess_variants(gray):
        raw = pytesseract.image_to_string(variant, config=_OCR_CONFIG).strip()
        if not raw:
            continue
        _, s = _best_fuzzy(raw, all_names)
        if s > best_score:
            best_score, best_raw = s, raw
    ocr_ms = (time.perf_counter() - t_ocr) * 1000

    best_name, score = _best_fuzzy(best_raw, all_names)
    return {
        "slot":           slot,
        "raw":            best_raw,
        "match":          best_name if score >= threshold else None,
        "score":          round(score, 2),
        "best_candidate": best_name,
        "crop_ms":        round(crop_ms, 1),
        "ocr_ms":         round(ocr_ms, 1),
    }


def ocr_all_slots(name_regions: list, threshold: float) -> list[dict]:
    """
    Read all 5 shop slots from the live screen.
    Single ImageGrab for the whole band + parallel per-slot Tesseract calls.
    Each result includes crop_ms and ocr_ms timing.
    Returns list of {slot, raw, match, score, best_candidate, crop_ms, ocr_ms}.
    """
    if not OCR_AVAILABLE:
        return []

    pad       = 4
    xs        = [r[0] for r in name_regions]
    ys        = [r[1] for r in name_regions]
    min_y     = min(ys)
    band_x0   = min(xs)
    band_y0   = min_y - pad
    band_x1   = max(r[0] + r[2] for r in name_regions)
    band_y1   = max(r[1] + r[3] for r in name_regions) + pad

    full_rgb  = np.array(ImageGrab.grab(bbox=(band_x0, band_y0, band_x1, band_y1)))
    all_names = [n for names in TFT_UNITS.values() for n in names]

    def _ocr_slot(args):
        i, (rx, ry, rw, rh) = args
        t_crop = time.perf_counter()
        y0   = ry - band_y0
        crop = full_rgb[y0 - pad : y0 + rh + pad, rx - band_x0 : rx - band_x0 + rw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop_ms = (time.perf_counter() - t_crop) * 1000
        return _ocr_gray(gray, i + 1, threshold, all_names, crop_ms)

    with ThreadPoolExecutor(max_workers=len(name_regions)) as pool:
        results = list(pool.map(_ocr_slot, enumerate(name_regions)))
    return results


def ocr_from_image_file(img_path: str, name_regions: list,
                         threshold: float) -> list[dict]:
    """
    Run OCR on a saved screenshot file.
    Auto-scales name_regions from 1920×1080 to the image dimensions.
    Uses the same _ocr_gray core as ocr_all_slots.
    Returns list of {slot, raw, match, score, best_candidate, crop_ms, ocr_ms,
                     img_size, scaled_region}.
    """
    if not OCR_AVAILABLE:
        return []

    from PIL import Image as _Image
    img_pil      = _Image.open(img_path).convert("RGB")
    img_w, img_h = img_pil.size
    img_cv       = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    REF_W, REF_H = 1920, 1080
    sx, sy       = img_w / REF_W, img_h / REF_H
    all_names    = [n for names in TFT_UNITS.values() for n in names]
    results      = []

    for i, (rx, ry, rw, rh) in enumerate(name_regions):
        x  = int(rx * sx);  y  = int(ry * sy)
        w  = int(rw * sx);  h  = int(rh * sy)
        y0 = max(0, y);     y1 = min(img_h, y + h)

        t_crop = time.perf_counter()
        crop   = img_cv[y0:y1, x:x + w]
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_ms = (time.perf_counter() - t_crop) * 1000

        result = _ocr_gray(gray, i + 1, threshold, all_names, crop_ms)
        result["img_size"]      = (img_w, img_h)
        result["scaled_region"] = [x, y, w, h]
        results.append(result)

    return results


def fuzzy_match(ocr_text: str, wanted: list, threshold: float) -> str | None:
    """Return the best-matching name from `wanted`, or None if below threshold."""
    if not ocr_text:
        return None
    best_name, score = _best_fuzzy(ocr_text, wanted)
    return best_name if score >= threshold else None


# ── Worker thread ─────────────────────────────────────────────────────────────
from PyQt5.QtCore import QThread, pyqtSignal


class EscWatcher(QThread):
    """
    Dedicated thread that polls the ESC key independently of the roll loop.
    Emits `triggered` the moment ESC is held, then exits.
    Connect `triggered` to worker.stop() for instant cancellation regardless
    of what state the roll loop is in (sleeping, buying, waiting for D, etc.).
    """
    triggered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self) -> None:
        # Poll every 5 ms — unblocks immediately when stop() is called
        while not self._stop_event.wait(0.005):
            if _esc_pressed():
                self.triggered.emit()
                break   # one-shot: fire once and exit

    def stop(self) -> None:
        self._stop_event.set()


class RollWorker(QThread):
    status_signal = pyqtSignal(str)
    roll_signal   = pyqtSignal(int)
    found_signal  = pyqtSignal(str)
    shop_signal   = pyqtSignal(list)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg         = cfg
        self._stop_event = threading.Event()
        self._reason     = "Stopped."

    @property
    def _running(self) -> bool:
        return not self._stop_event.is_set()

    def _sleep(self, seconds: float) -> bool:
        """
        Sleep up to `seconds` while polling ESC every 10 ms.
        Returns True as soon as stop() is called or ESC is pressed.
        This means ESC is always responsive during any wait, with no extra thread.
        """
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
            # ── Trigger ───────────────────────────────────────────
            if auto_roll:
                # Auto: previous buy cycle is done → press D immediately
                if not _focus_tft():
                    self.stop("⚠ TFT window not found. Is the game running?")
                    break
                _press("d")
                count += 1
                self.roll_signal.emit(count)
            else:
                # Manual: wait for user to press D
                self.status_signal.emit("Waiting for D key press…")
                while self._running:
                    if _d_pressed():
                        break
                    self._sleep(0.015)
                if not self._running:
                    break
                # Wait for key release to avoid re-triggering
                while self._running and _d_pressed():
                    self._sleep(0.015)
                if not self._running:
                    break
                count += 1
                self.roll_signal.emit(count)
                if not _focus_tft():
                    self.stop("⚠ TFT window not found. Is the game running?")
                    break

            if not self._running:
                break

            # ── OCR (in BOT mode runs parallel with shop_wait) ────
            t0 = time.perf_counter()
            if bot_mode:
                fut = ThreadPoolExecutor(max_workers=1).submit(
                    ocr_all_slots, cfg["name_regions"], cfg["ocr_threshold"])
                self._sleep(cfg["shop_wait"])   # exits early on stop
                if not self._running:
                    break
                results = fut.result()         # blocks only if OCR not done yet
            else:
                if self._sleep(cfg["shop_wait"]):
                    break
                results = ocr_all_slots(cfg["name_regions"], cfg["ocr_threshold"])
            ocr_ms = (time.perf_counter() - t0) * 1000
            self.shop_signal.emit(results)

            # ── Buy matched slots ─────────────────────────────────
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

            bought_str = ", ".join(bought) if bought else "none"
            ocr_tag    = f"  [{ocr_ms:.0f} ms]"
            self.status_signal.emit(
                f"Roll {count}{ocr_tag}  |  bought: {bought_str}")

        self.status_signal.emit(self._reason)

    def stop(self, reason: str = "Stopped.") -> None:
        self._reason = reason
        self._stop_event.set()
