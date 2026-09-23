@echo off
setlocal
cd /d "%~dp0"
set PORT=5551

echo Starting local server for Stencil Generator...
start "Stencil Generator Server - close this window to stop" cmd /k npx --yes serve -l %PORT% .

timeout /t 3 /nobreak >nul
start "" http://localhost:%PORT%/

endlocal
