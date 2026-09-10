@echo off
REM ============================================================
REM  CarLaneI - ONE-CLICK Web App (frontend + backend)
REM
REM  Starts the FastAPI server (which serves the web frontend)
REM  and opens your browser to it. Everything runs from here.
REM
REM  Just double-click this file.
REM  Close this window (or press Ctrl+C) to stop the server.
REM ============================================================

set PYTHONIOENCODING=utf-8
set ROOT=D:\carLane
set PY=%ROOT%\.venv\Scripts\python.exe
set PORT=8000
set URL=http://127.0.0.1:%PORT%/

cd /d %ROOT%

REM --- If something is already on the port, just open the browser to it ---
netstat -ano | findstr /r /c:"[:.]%PORT% .*LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo CarLaneI already running on %URL%
    start "" %URL%
    echo.
    echo This window can be closed. The server is in its own window.
    pause
    goto :eof
)

echo ============================================================
echo   CarLaneI - starting web app on %URL%
echo ============================================================
echo.
echo   Loading models (first start takes ~15s)...
echo   Your browser will open automatically.
echo   Leave THIS window open while you use the app.
echo   Close it (or Ctrl+C) to stop.
echo.

REM Open the browser a few seconds after the server begins booting
start "" cmd /c "timeout /t 7 >nul & start %URL%"

REM Run the server in THIS window (so logs are visible, closing it stops it)
"%PY%" -m uvicorn web.backend:app --host 127.0.0.1 --port %PORT%

echo.
echo   Server stopped.
pause
