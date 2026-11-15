@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 --version >nul 2>nul
        if not errorlevel 1 (
            py -3.12 -m venv .venv
            set "PYTHON=.venv\Scripts\python.exe"
        )
    )
    if not defined PYTHON (
        where python >nul 2>nul
        if errorlevel 1 (
            echo Khong tim thay Python. Hay cai Python 3.12 va chon Add Python to PATH.
            pause
            exit /b 1
        )
        python -m venv .venv
        set "PYTHON=.venv\Scripts\python.exe"
    )
)

"%PYTHON%" -c "import flask, rembg, cv2, PIL, numpy" >nul 2>nul
if errorlevel 1 (
    echo Dang cai dat thu vien lan dau, vui long cho...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto install_error
    "%PYTHON%" -m pip install -r "BackgroundChanger\requirements.txt"
    if errorlevel 1 goto install_error
)

echo Dang khoi dong Background Studio...
start "Background Studio" http://127.0.0.1:5000
"%PYTHON%" "BackgroundChanger\app.py"
pause
exit /b

:install_error
echo Cai dat that bai. Hay kiem tra ket noi Internet va thu chay lai tep nay.
pause
exit /b 1
