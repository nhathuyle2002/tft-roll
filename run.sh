#!/usr/bin/env bash
echo "=== TFT Roll Tool — Set 17: Space Gods ==="
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

# Check Tesseract (optional – Smart Buy needs it)
if ! command -v tesseract &>/dev/null; then
    echo "[WARNING] Tesseract OCR not found."
    echo "  Smart Buy (auto-identify units) will be disabled."
    echo "  To enable it later, install Tesseract:"
    echo "    macOS:  brew install tesseract"
    echo "    Ubuntu: sudo apt install tesseract-ocr"
    echo
fi

echo "Installing / verifying Python packages..."
# NOTE: the module is 'import cv2' but pip package is 'opencv-python'
pip3 install PyQt5 pyautogui pydirectinput pytesseract opencv-python numpy Pillow --quiet 2>&1
echo "Done."
echo

echo "Launching TFT Roll Tool..."
python3 tft_roll_tool.py
