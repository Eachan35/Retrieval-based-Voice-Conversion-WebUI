@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "RVC_ENV=D:\AppData\anaconda3\envs\rvc"
set "PATH=%RVC_ENV%;%RVC_ENV%\Scripts;%PATH%"
echo Starting RVC. Please wait...
"%RVC_ENV%\python.exe" realtime_gui.py
pause
