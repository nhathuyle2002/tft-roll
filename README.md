# TFT Roll Tool — Set 17: Space Gods

An automation tool for **Teamfight Tactics** that watches your shop, recognises unit names via OCR or hash lookup, and automatically buys the units you've selected for your build.

---

## Introduction

### What it does

- **Build board** — pick the units you want from the full Set 17 roster (62 units, costs 1–5). The tool only buys those units and ignores everything else.
- **Auto Roll** — optionally let the script press **D** for you between buy cycles, or play manually and have the tool react to your own D presses.
- **Human / BOT mode** — two buy-speed presets. Human uses natural timing; BOT runs OCR in parallel with the shop-load wait for maximum throughput.
- **ESC to stop** — pressing Escape stops the tool instantly at any point during the loop (no dedicated thread; polled every 10 ms inside every wait).
- **Log overlay** — a semi-transparent always-on-top window shows live roll count and OCR results without alt-tabbing back to the tool.
- **Hash-first recognition (v2)** — after a unit is seen once, its name region is hashed. On every subsequent roll that slot is recognised instantly with zero Tesseract calls.
- **Train Mode (v2)** — upload screenshots or capture the live screen to build the hash map. Manual and timed auto-capture modes included.

### Platform

Windows only. Requires DirectInput keyboard support and Win32 mouse APIs (both Vanguard-compatible).

---

## Installation

### 1 — Python

Python **3.10 or later** is required.

```bash
python --version
```

### 2 — Tesseract OCR binary

Download and install the Tesseract binary from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) before installing the Python packages.

### 3 — Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run

```bash
run.bat
```

Or directly:

```bash
python tft_roll_tool.py
```

---

## Usage

1. Launch the tool and select units on the **Build & Roll** tab.
2. Configure timing in the **Settings** tab if needed (defaults target 1920×1080 fullscreen).
3. Switch to TFT, then press **START** (or keep TFT focused and let the countdown finish).
4. Press **ESC** or click **STOP** at any time to halt.

---

## Project structure

```
tft_roll_tool.py      # UI — single entry point, all tabs
tft_backend.py        # v1 backend: input layer, OCR pipeline, RollWorker
tft_v2_backend.py     # v2 backend: HashMapper, hash-first lookup, RollWorkerV2, AutoCaptureWorker
hashmap.json          # persisted hash → unit name map (auto-created)
train/                # training images and result text files (auto-created)
requirements.txt
run.sh / run.bat
```

---

## Version history

### v2 — Hash-first recognition + Train Mode

**New features**

- **Hash-first unit lookup** — on each roll, every name crop is normalised (Otsu binarise → morphological open → tight-trim to text bounding box → resize to 128×20) and MD5-hashed. If the hash is already in `hashmap.json`, the unit name is returned immediately — zero Tesseract calls for known units.
- **Cross-slot invariance** — the normalisation pipeline removes background colour differences, per-slot anti-aliasing noise, and text positional offsets, so the same unit in slot 1 and slot 5 produces an identical hash.
- **HashMapper** — thread-safe in-memory store backed by `hashmap.json`. Disk writes are async (via `ThreadPoolExecutor`) so they never block the roll loop.
- **Train Mode tab** — dedicated tab to build and inspect the hash map:
  - Upload any screenshot → Run Hash+OCR → results logged with `⚡` (hash hit) or `🔬` (OCR fallback).
  - **Manual capture** — captures the live screen once in a background thread.
  - **Auto capture** — captures every N seconds (configurable, default 15 s) while you play.
  - All captured images saved to `train/<n>_image.png` with corresponding `train/<n>_result.txt`.
- **Roll status** now shows hash hit count per roll: `[42ms ⚡4/5]`.
- **`RollWorkerV2`** — drop-in replacement for `RollWorker` using `ocr_all_slots_v2` (hash-first).

**Improvements**

- `EscWatcher` dedicated thread removed — ESC is polled every 10 ms inside `_sleep()`, keeping the worker thread count minimal.
- All `threading.Thread` fire-and-forget calls replaced with a shared `ThreadPoolExecutor` (`_async_pool`).

---

### v1 — Initial release

- PyQt5 UI with **Build & Roll**, **Settings**, and **OCR Test** tabs.
- Full Set 17 unit roster (62 units, costs 1–5).
- Tesseract OCR with 6 preprocessing variants (Otsu + adaptive at 3× and 4× scale), PSM 7.
- Fuzzy matching via `difflib.SequenceMatcher`, configurable threshold (default 0.50).
- Parallel slot OCR using `ThreadPoolExecutor(max_workers=5)` with a single `ImageGrab` per roll.
- **Auto Roll** mode (script presses D) and **Manual** mode (waits for user D press).
- **Human** and **BOT** buy-speed presets.
- DirectInput keyboard (`pydirectinput`) — Vanguard-safe.
- `SetCursorPos + mouse_event` click API — legacy Win32 path, different hook from `SendInput`.
- Semi-transparent always-on-top **Log Overlay** (560×420, 500-line buffer, drag-to-move).
- macOS/Linux no-op input layer — full UI usable for testing without Windows.
- Backend/UI split: `tft_backend.py` (all logic) and `tft_roll_tool.py` (pure UI).
