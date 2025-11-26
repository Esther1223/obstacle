@echo off
setlocal EnableDelayedExpansion

title VIVE Connect WiFi Sync (Do not close this window)

echo.
echo =====================================================
echo   VIVE Connect WiFi Sync is running...
echo =====================================================
echo.

::=== 這裡改成你的手機 IP（不要多打空白） ===
set DEVICE_IP=172.26.5.8

::=== 手機與電腦資料夾 ===
set PC_DIR=C:\Users\USER\Desktop\obstacle\backup

::=== 建立備份資料夾（若不存在） ===
if not exist "%PC_DIR%" (
    echo Creating backup directory: %PC_DIR%
    mkdir "%PC_DIR%"
)

echo -----------------------------------------------------
echo [1] activate ADB server
echo -----------------------------------------------------
adb kill-server >nul 2>&1
adb start-server >nul 2>&1

echo.
echo -----------------------------------------------------
echo [2] try to connect to phone %DEVICE_IP%:5555 via Wi-Fi
echo -----------------------------------------------------
adb connect %DEVICE_IP%:5555

echo.
echo Currently detected device list:
adb devices
echo.

:: 檢查是否有任何 device 狀態
set DEV_STATE=
for /f "skip=1 tokens=1,2" %%A in ('adb devices') do (
    if NOT "%%A"=="" (
        set DEV_STATE=%%B
        goto :CHK_STATE_DONE
    )
)

:CHK_STATE_DONE

if NOT "!DEV_STATE!"=="device" (
    echo.
    echo [ERROR] Cannot connect to device via Wi-Fi.
    echo Please check:
    echo    1. The phone and computer are on the same Wi-Fi network
    echo    2. USB debugging is enabled on the phone
    echo    3. The phone has authorized USB debugging for this computer
    echo.
    pause
    exit /b
)

echo -----------------------------------------------------
echo [3] list files in /sdcard/DCIM/VIVE Connect
echo -----------------------------------------------------
adb shell "ls '/sdcard/DCIM/VIVE Connect'" > file_list.txt

set fileCount=0
for /f "usebackq delims=" %%F in ("file_list.txt") do (
    if NOT "%%F"=="" (
        set /a fileCount+=1
        echo Found: %%F
    )
)

if %fileCount%==0 (
    echo.
    echo -----------------------------------------------------
    echo [Info] No new files to sync
    echo -----------------------------------------------------
    del file_list.txt
    echo.
    pause
    exit /b
)

echo.
echo -----------------------------------------------------
echo [4] start syncing and deleting files on phone
echo -----------------------------------------------------
for /f "usebackq delims=" %%F in ("file_list.txt") do (
    if NOT "%%F"=="" (
        echo Pulling: %%F
        adb pull "/sdcard/DCIM/VIVE Connect/%%F" "%PC_DIR%"
        echo Deleting from phone: %%F
        adb shell "rm '/sdcard/DCIM/VIVE Connect/%%F'"
        echo.
    )
)

del file_list.txt


echo.
echo =====================================================
echo      Sync complete! Files saved to:
echo        %PC_DIR%
echo =====================================================
echo.

pause
endlocal
