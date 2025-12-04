@echo off
setlocal enabledelayedexpansion

REM Change to repo root
cd /d "%~dp0"

set MODE=%1

REM Activate conda env (adjust path/env name if different)
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" obstacle

if "%MODE%"=="--watch" (
    python "%~dp0obstacle.py" --watch
    endlocal
    exit /b
)

REM 1) Sync photos
call "%~dp0sync_vive_connect.bat"

REM 2) Run obstacle detection
python "%~dp0obstacle.py"

endlocal
