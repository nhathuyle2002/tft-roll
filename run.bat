@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================
echo   TFT Roll Tool - Set 17: Space Gods
echo ================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo.
    echo   Install Python 3.10+ from https://python.org
    echo   During install: check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v
echo.

:: ── Check Tesseract OCR (optional) ───────────────────────────
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Tesseract OCR not found.
    echo   Smart Buy will be disabled.
    echo   To enable it: https://github.com/UB-Mannheim/tesseract/wiki
    echo.
) else (
    echo   [OK] Tesseract found.
)

:: ── Install Python packages ───────────────────────────────────
echo Installing / verifying Python packages...
echo.

pip install ^
    PyQt5>=5.15 ^
    pyautogui>=0.9.54 ^
    pydirectinput>=1.0.4 ^
    pytesseract>=0.3.10 ^
    opencv-python>=4.8 ^
    numpy>=1.24 ^
    Pillow>=10.0 ^
    --quiet --no-warn-script-location

if errorlevel 1 (
    echo.
    echo [WARN] Some packages may have failed. Trying with -r requirements.txt...
    pip install -r requirements.txt --quiet --no-warn-script-location
)

echo   [OK] Packages ready.
echo.

:: ── Launch ───────────────────────────────────────────────────
echo Launching TFT Roll Tool...
echo   (Close this window to stop the tool)
echo.
python tft_roll_tool.py

echo.
echo TFT Roll Tool exited.
pause
