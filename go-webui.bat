@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "RVC_ENV=D:\AppData\anaconda3\envs\rvc"
set "PATH=%RVC_ENV%;%RVC_ENV%\Scripts;%PATH%"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=localhost,127.0.0.1,::1,%NO_PROXY%"
echo 首次启动程序可能需要耐心等待20秒
"%RVC_ENV%\python.exe" webui.py --pycmd "%RVC_ENV%\python.exe" --port 7865
pause
