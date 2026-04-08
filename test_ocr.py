"""
OCR debug tool — point it at one or more TFT screenshots to see what
Tesseract reads from each shop slot, compared against a ground-truth
.txt file (one unit name per line, same base name as the image).

Usage:
    python test_ocr.py test_1.png test_2.png
    python test_ocr.py *.png
"""

import sys
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

# ── Pull shared constants & helpers from the backend (single source of truth) ─
from tft_backend import (
    TFT_UNITS,
    DEFAULTS,
    _OCR_WL,
    _preprocess_variants,
    _best_fuzzy,
)

# Slot regions come straight from DEFAULTS so test_ocr and the live tool
# always scan the exact same coordinates.
REF_W, REF_H = 1920, 1080
SLOT_REGIONS  = DEFAULTS["name_regions"]   # [[x, y, w, h], ...]

# Extra PSM configs used only for the debug table (backend uses PSM 7 only)
OCR_CONFIGS = [
    ("psm7 (line)",  f'--psm 7  --oem 1 -c "tessedit_char_whitelist={_OCR_WL}"'),
    ("psm8 (word)",  f'--psm 8  --oem 1 -c "tessedit_char_whitelist={_OCR_WL}"'),
    ("psm13 (raw)",  f'--psm 13 --oem 1 -c "tessedit_char_whitelist={_OCR_WL}"'),
]

MATCH_THRESHOLD = 0.50

# Flat unit list derived from the backend roster (no duplication)
TFT_ALL_UNITS = [n for names in TFT_UNITS.values() for n in names]

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "ocr_debug")
# ─────────────────────────────────────────────────────────────────────────────


def scale_regions(img_w: int, img_h: int) -> list:
    """Scale SLOT_REGIONS from 1920×1080 to the actual image dimensions."""
    sx, sy = img_w / REF_W, img_h / REF_H
    return [[int(x*sx), int(y*sy), int(w*sx), int(h*sy)]
            for x, y, w, h in SLOT_REGIONS]


def crop_name(img_cv: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
    """Return the fixed name-region crop and its absolute y bounds."""
    img_h = img_cv.shape[0]
    y0 = max(0, y)
    y1 = min(img_h, y + h)
    return img_cv[y0:y1, x:x + w], y0, y1


_VARIANT_NAMES = [
    "otsu_inv_3x", "otsu_norm_3x", "adaptive_3x",
    "otsu_inv_4x", "otsu_norm_4x", "adaptive_4x",
]

def preprocess_variants(gray):
    """Return [(name, img), ...] — wraps the backend's unnamed version."""
    return list(zip(_VARIANT_NAMES, _preprocess_variants(gray)))


# ── Output image helpers ─────────────────────────────────────────────────────

def save_annotated_image(img_cv: np.ndarray, slot_info: list,
                         out_path: str) -> None:
    """
    Draw a labelled bounding box on the original screenshot for every detected
    name region and save to out_path.
    slot_info: list of dicts with keys x, w, abs_y0, abs_y1, label, passed.
    """
    vis = img_cv.copy()
    for s in slot_info:
        x, w       = s["x"], s["w"]
        y0, y1     = s["abs_y0"], s["abs_y1"]
        label      = s["label"]
        passed     = s.get("passed")            # True / False / None
        color      = (86, 211, 100) if passed else ((81, 149, 255) if passed is None else (81, 81, 248))

        # Box
        cv2.rectangle(vis, (x, y0), (x + w, y1), color, 2)
        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = y0 - 6
        cv2.rectangle(vis, (x, ly - th - 4), (x + tw + 4, ly + 2), color, cv2.FILLED)
        cv2.putText(vis, label, (x + 2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)


def save_name_strip(crops: list, slot_labels: list, out_path: str) -> None:
    """
    Stitch all 5 name crops side-by-side (padded to the same height) into one
    image and write it to out_path.  Each crop gets a white label bar on top.
    crops:       list of BGR numpy arrays (one per slot).
    slot_labels: list of strings shown above each crop.
    """
    LABEL_H  = 18
    PAD      = 4            # horizontal gap between slots
    SCALE    = 3            # upscale each crop so text is legible

    scaled = [cv2.resize(c, None, fx=SCALE, fy=SCALE,
                         interpolation=cv2.INTER_CUBIC) for c in crops]
    max_h  = max(c.shape[0] for c in scaled)
    total_w = sum(c.shape[1] for c in scaled) + PAD * (len(scaled) - 1)

    canvas = np.full((LABEL_H + max_h, total_w, 3), 30, dtype=np.uint8)  # dark bg

    cx = 0
    for i, (crop, lbl) in enumerate(zip(scaled, slot_labels)):
        h, w = crop.shape[:2]
        # Crop
        canvas[LABEL_H:LABEL_H + h, cx:cx + w] = crop
        # Label bar
        cv2.rectangle(canvas, (cx, 0), (cx + w, LABEL_H), (50, 50, 50), cv2.FILLED)
        cv2.putText(canvas, lbl, (cx + 3, LABEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cx += w + PAD

    cv2.imwrite(out_path, canvas)


def load_expected(image_path: str) -> list:
    """Load ground-truth names from a .txt file next to the image, if it exists."""
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fuzzy_score(a: str, b: str) -> float:
    """Pairwise fuzzy score — delegates to the backend's matcher."""
    return _best_fuzzy(a, [b])[1]


def run_test(image_path: str) -> dict:
    """
    Run OCR on all 5 shop slots for one image.
    Returns {'slots': [...], 'correct': int, 'total': int}.
    """
    expected = load_expected(image_path)
    has_gt   = len(expected) > 0

    print(f"\n{'='*62}")
    print(f" OCR Test : {os.path.basename(image_path)}")
    if has_gt:
        print(f" Expected : {'  |  '.join(expected)}")
    print(f"{'='*62}")

    img_pil = Image.open(image_path).convert("RGB")
    img_w, img_h = img_pil.size
    print(f" Image size: {img_w} × {img_h}")

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    regions = scale_regions(img_w, img_h)
    print(f" Regions scaled from {REF_W}×{REF_H} reference")

    os.makedirs(DEBUG_DIR, exist_ok=True)
    img_base = os.path.splitext(os.path.basename(image_path))[0]

    slot_results    = []
    crops_for_strip = []   # raw name crops collected for the strip image
    annot_info      = []   # per-slot metadata for the annotated screenshot

    for slot_idx, (x, y, w, h) in enumerate(regions):
        slot_num = slot_idx + 1

        # ── Fixed name-text crop ──────────────────────────────────────────
        crop_bgr, abs_y0, abs_y1 = crop_name(img_cv, x, y, w, h)
        print(f"\n── Slot {slot_num}  x={x}  y={y}  w={w}  h={h}  "
              f"crop={crop_bgr.shape[1]}×{crop_bgr.shape[0]}px ──")

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crops_for_strip.append(crop_bgr)

        # Save the raw (auto-detected) crop
        crop_path = os.path.join(DEBUG_DIR, f"{img_base}_slot{slot_num}_raw.png")
        cv2.imwrite(crop_path, crop_bgr)
        print(f"   Saved raw crop → {crop_path}")

        # ── Try every variant × config; pick by best fuzzy match ──────────
        variants = preprocess_variants(gray)
        for var_name, var_img in variants:
            vpath = os.path.join(DEBUG_DIR, f"{img_base}_slot{slot_num}_{var_name}.png")
            cv2.imwrite(vpath, var_img)

        # best_raw  = the OCR string that produced the best match
        # best_unit = the unit name it matched to
        # best_score = fuzzy score
        best_raw, best_unit, best_score = "", "", 0.0

        print(f"   {'Config':<16} {'Variant':<20} {'OCR text':<28} {'→ unit (score)'}")
        print(f"   {'-'*16} {'-'*20} {'-'*28} {'-'*20}")
        for cfg_name, cfg in OCR_CONFIGS:
            for var_name, var_img in variants:
                text = pytesseract.image_to_string(var_img, config=cfg).strip()
                text = text.replace("\n", " ")
                if not text:
                    continue
                # Find best-matching unit for this read
                top_unit, top_score = _best_fuzzy(text, TFT_ALL_UNITS)
                marker = ""
                if top_score > best_score:
                    best_score, best_unit, best_raw = top_score, top_unit, text
                    marker = "  ◀ best"
                print(f"   {cfg_name:<16} {var_name:<20} '{text[:26]}'  "
                      f"→ {top_unit} ({top_score:.2f}){marker}")

        print(f"\n   ★ Slot {slot_num}: raw='{best_raw}'  matched='{best_unit}'  "
              f"score={best_score:.2f}")

        # ── Ground truth comparison ───────────────────────────────────────
        gt_name = expected[slot_idx] if has_gt and slot_idx < len(expected) else None
        # Use the higher of: (raw OCR vs ground truth) or (matched unit vs ground truth).
        # This lets units absent from the roster (e.g. from a newer patch) still pass
        # when the raw read is correct.
        if gt_name:
            raw_gt   = fuzzy_score(best_raw,  gt_name)
            unit_gt  = fuzzy_score(best_unit, gt_name)
            gt_score = max(raw_gt, unit_gt)
        else:
            gt_score = None
        passed   = gt_score is not None and gt_score >= MATCH_THRESHOLD

        if gt_name:
            mark = "✓" if passed else "✗"
            print(f"   ↳ Expected: '{gt_name}'  unit_score={gt_score:.2f}  {mark}")

        slot_results.append({
            "slot":      slot_num,
            "raw":       best_raw,
            "matched":   best_unit,
            "ocr_score": round(best_score, 2),
            "expected":  gt_name,
            "gt_score":  gt_score,
            "passed":    passed,
        })
        annot_info.append({
            "x": x, "w": w, "abs_y0": abs_y0, "abs_y1": abs_y1,
            "label": f"S{slot_num}: {best_unit or best_raw or '?'}",
            "passed": passed,
        })

    # ── Per-image summary ─────────────────────────────────────────────────
    correct = sum(1 for s in slot_results if s["passed"])
    total   = sum(1 for s in slot_results if s["expected"] is not None)


    if has_gt:
        pct = (correct / total * 100) if total else 0
        print(f"\n  {'─'*56}")
        print(f"  Result: {correct}/{total} slots correct  ({pct:.0f}%)")
        for s in slot_results:
            mark = "✓" if s["passed"] else ("✗" if s["expected"] else "—")
            exp_part = (f"  exp='{s['expected']}'  unit_score={s['gt_score']:.2f}"
                        if s["expected"] else "")
            print(f"    Slot {s['slot']}: {mark}  matched='{s['matched']}'"
                  f"  (raw='{s['raw']}'){exp_part}")

    # ── Save output images ────────────────────────────────────────────────
    annot_path = os.path.join(DEBUG_DIR, f"{img_base}_annotated.png")
    save_annotated_image(img_cv, annot_info, annot_path)
    print(f"  Annotated screenshot → {annot_path}")

    strip_labels = [
        f"S{s['slot']}:{s['matched'] or s['raw'] or '?'}" for s in slot_results
    ]
    strip_path = os.path.join(DEBUG_DIR, f"{img_base}_name_strip.png")
    save_name_strip(crops_for_strip, strip_labels, strip_path)
    print(f"  Name-crop strip      → {strip_path}")

    print(f"\n  Debug crops → {DEBUG_DIR}/")
    print(f"{'='*62}\n")

    return {"slots": slot_results, "correct": correct, "total": total}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <screenshot.png> [more.png ...]")
        sys.exit(1)

    grand_correct = grand_total = 0
    for path in sys.argv[1:]:
        result = run_test(path)
        grand_correct += result["correct"]
        grand_total   += result["total"]

    if len(sys.argv) > 2 and grand_total > 0:
        pct = grand_correct / grand_total * 100
        print(f"{'#'*62}")
        print(f" OVERALL  {grand_correct}/{grand_total} slots correct  ({pct:.0f}%)")
        print(f"{'#'*62}\n")
