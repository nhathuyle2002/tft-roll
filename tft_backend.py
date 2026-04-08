"""
tft_backend.py — Game data, input layer, OCR helpers, worker thread.
Imported by tft_roll_tool.py (UI) and usable standalone / in tests.
"""

import time
import difflib
import ctypes
from concurrent.futures import ThreadPoolExecutor

import pydirectinput
pydirectinput.FAILSAFE = False

# ── Win32 helpers ─────────────────────────────────────────────────────────────
_user32 = ctypes.windll.user32

_VK_ESCAPE = 0x1B
_VK_D      = 0x44   # 'D' virtual key code

def _d_pressed() -> bool:
    """True while D is held down (high-order bit set)."""
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
    for title in _TFT_TITLES:
        hwnd = _user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
    return 0


def _focus_tft() -> None:
    """Bring the TFT window to the foreground so clicks register."""
    hwnd = _find_tft_hwnd()
    if hwnd:
        _user32.SetForegroundWindow(hwnd)
        time.sleep(0.05)   # let the OS settle focus


def _esc_pressed() -> bool:
    try:
        return bool(_user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)
    except Exception:
        return False


def _press(key: str) -> None:
    pydirectinput.press(key)


# Mouse movement: SetCursorPos (no SendInput, not intercepted by Vanguard)
# Mouse click:   legacy mouse_event API (different hook path from SendInput)
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004


def _click(x: int, y: int) -> None:
    """
    SetCursorPos  → moves cursor without going through SendInput at all.
    mouse_event   → legacy Win32 click API (older than SendInput, different
                    Vanguard hook path than pydirectinput/pyautogui).
    SetForegroundWindow is called once per cycle before this (in the worker).
    """
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


# ── Defaults (1920×1080 fullscreen) ──────────────────────────────────────────
DEFAULTS = {
    # Timing
    "roll_delay":    1.5,   # seconds between each D press
    "pre_delay":     3,     # countdown before automation starts
    "shop_wait":     0.5,   # seconds to wait after rolling for shop to load
    "buy_delay":     0.12,  # seconds between clicking each shop slot

    # OCR
    "ocr_threshold": 0.50,

    # Click centers — each shop card (x, y)
    "click_pos": [
        [575,  992],
        [775,  992],
        [975,  992],
        [1175, 992],
        [1375, 992],
    ],
    # Name-text crop regions [x, y, w, h] — calibrated from annotated screenshots
    "name_regions": [
        [480,  1045, 145, 20],
        [685,  1045, 145, 20],
        [885,  1045, 145, 20],
        [1085, 1045, 145, 20],
        [1290, 1045, 145, 20],
    ],
}


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

        # ── crop timing ──────────────────────────────────────
        t_crop = time.perf_counter()
        y0   = ry - band_y0
        crop = full_rgb[y0 - pad : y0 + rh + pad, rx - band_x0 : rx - band_x0 + rw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop_ms = (time.perf_counter() - t_crop) * 1000

        # ── OCR timing ───────────────────────────────────────
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
            "slot":           i + 1,
            "raw":            best_raw,
            "match":          best_name if score >= threshold else None,
            "score":          round(score, 2),
            "best_candidate": best_name,
            "crop_ms":        round(crop_ms, 1),
            "ocr_ms":         round(ocr_ms, 1),
        }

    with ThreadPoolExecutor(max_workers=len(name_regions)) as pool:
        results = list(pool.map(_ocr_slot, enumerate(name_regions)))
    return results


def ocr_from_image_file(img_path: str, name_regions: list,
                         threshold: float) -> list[dict]:
    """
    Run OCR on a saved screenshot file.
    Auto-scales name_regions from 1920×1080 to the image dimensions.
    Returns list of {slot, raw, match, score, best_candidate, img_size, scaled_region}.
    """
    if not OCR_AVAILABLE:
        return []

    from PIL import Image as _Image
    img_pil          = _Image.open(img_path).convert("RGB")
    img_w, img_h     = img_pil.size
    img_cv           = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    REF_W, REF_H     = 1920, 1080
    sx, sy           = img_w / REF_W, img_h / REF_H
    all_names        = [n for names in TFT_UNITS.values() for n in names]
    results          = []

    for i, (rx, ry, rw, rh) in enumerate(name_regions):
        x   = int(rx * sx);  y  = int(ry * sy)
        w   = int(rw * sx);  h  = int(rh * sy)
        y0  = max(0, y);     y1 = min(img_h, y + h)

        crop = img_cv[y0:y1, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        best_raw, best_score = "", 0.0
        for variant in _preprocess_variants(gray):
            raw = pytesseract.image_to_string(variant, config=_OCR_CONFIG).strip()
            if not raw:
                continue
            _, score = _best_fuzzy(raw, all_names)
            if score > best_score:
                best_score = score
                best_raw   = raw

        best_name, score = _best_fuzzy(best_raw, all_names)
        results.append({
            "slot":           i + 1,
            "raw":            best_raw,
            "match":          best_name if score >= threshold else None,
            "score":          round(score, 2),
            "best_candidate": best_name,
            "img_size":       (img_w, img_h),
            "scaled_region":  [x, y, w, h],
        })

    return results


def fuzzy_match(ocr_text: str, wanted: list, threshold: float) -> str | None:
    """Return the best-matching name from `wanted`, or None if below threshold."""
    if not ocr_text:
        return None
    best_name, score = _best_fuzzy(ocr_text, wanted)
    return best_name if score >= threshold else None


# ── Worker thread ─────────────────────────────────────────────────────────────
from PyQt5.QtCore import QThread, pyqtSignal


class RollWorker(QThread):
    status_signal = pyqtSignal(str)
    roll_signal   = pyqtSignal(int)
    found_signal  = pyqtSignal(str)
    shop_signal   = pyqtSignal(list)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg      = cfg
        self._running = False

    def run(self) -> None:
        self._running = True
        cfg      = self.cfg
        count    = 0
        reason   = "Stopped."
        listen_d = cfg.get("listen_d", True)   # True = wait for user D press (default)
        auto_roll = cfg.get("auto_roll", False)

        self.status_signal.emit(f"Starting in {cfg['pre_delay']}s – switch to TFT!")
        time.sleep(cfg["pre_delay"])

        if listen_d:
            self.status_signal.emit("Waiting for D key press…")

        while self._running:
            # ── ESC = instant stop ────────────────────────────────
            if _esc_pressed():
                reason = "Stopped by ESC."
                break

            # ── Trigger: wait for D press OR enter auto scan ──────
            if listen_d:
                # Wait until D goes down
                while self._running:
                    if _esc_pressed():
                        reason = "Stopped by ESC."
                        self._running = False
                        break
                    if _d_pressed():
                        break
                    time.sleep(0.015)
                if not self._running:
                    break
                # Wait for D to be released (avoid re-triggering while held)
                while self._running and _d_pressed():
                    time.sleep(0.015)
                count += 1
                self.roll_signal.emit(count)
                # Let the shop finish loading
                time.sleep(cfg["shop_wait"])

            # ── OCR + buy ─────────────────────────────────────────
            if _esc_pressed():
                reason = "Stopped by ESC."
                break
            _focus_tft()
            t0 = time.perf_counter()

            results = ocr_all_slots(cfg["name_regions"], cfg["ocr_threshold"])
            ocr_ms  = (time.perf_counter() - t0) * 1000
            self.shop_signal.emit(results)

            bought = []
            for r, pos in zip(results, cfg["click_pos"]):
                if not self._running:
                    break
                if r["match"]:
                    _click(pos[0], pos[1])
                    bought.append(r["match"])
                    self.found_signal.emit(
                        f"Slot {r['slot']} → {r['match']} ✓  ('{r['raw']}')")
                    time.sleep(cfg["buy_delay"])

            bought_str = ", ".join(bought) if bought else "none"
            ocr_tag    = f"  [{ocr_ms:.0f} ms]" if ocr_ms else ""

            # ── Roll / wait ───────────────────────────────────────
            if not listen_d and auto_roll and self._running:
                _press("d")
                count += 1
                self.roll_signal.emit(count)
                self.status_signal.emit(
                    f"Roll {count}{ocr_tag}  |  bought: {bought_str}")
                elapsed, step = 0.0, 0.05
                while elapsed < cfg["roll_delay"] and self._running:
                    if _esc_pressed():
                        reason = "Stopped by ESC."
                        self._running = False
                        break
                    time.sleep(step)
                    elapsed += step
            elif listen_d:
                self.status_signal.emit(
                    f"Roll {count}{ocr_tag}  |  bought: {bought_str}")
                self.status_signal.emit("Waiting for D key press…")
            else:
                # auto scan without auto roll — wait shop_wait then re-scan
                self.status_signal.emit(
                    f"Scan{ocr_tag}  |  bought: {bought_str}")
                elapsed, step = 0.0, 0.05
                while elapsed < cfg["shop_wait"] and self._running:
                    if _esc_pressed():
                        reason = "Stopped by ESC."
                        self._running = False
                        break
                    time.sleep(step)
                    elapsed += step

        self.status_signal.emit(reason)

    def stop(self) -> None:
        self._running = False
