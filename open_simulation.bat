@echo off
echo ========================================================
echo   Launching Autonomous 3D Real-Time Simulation Cockpit
echo ========================================================
echo.

:: Detect local network IPv4 address
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' }).IPAddress | Select-Object -First 1"') do set "NET_IP=%%i"

:: Check if server is already running on port 8000
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting Python simulation server on port 8000 ...
    start /B python -u server.py >nul 2>&1
    timeout /t 3 /nobreak >nul
) else (
    echo Simulation server is already running.
)

echo.
echo ========================================================
echo   Server is live!
echo   * Local URL:   http://localhost:8000
echo   * Network URL: http://%NET_IP%:8000
echo ========================================================
echo.
echo Opening browser to http://localhost:8000 ...
start "" "http://localhost:8000"
echo.
pause
