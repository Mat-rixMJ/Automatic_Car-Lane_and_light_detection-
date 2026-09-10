@echo off
REM ============================================================
REM  CarLaneI Web Showcase - one-click launcher
REM  Starts the FastAPI backend (serves the frontend too) and
REM  opens the browser. Ctrl+C in this window to stop.
REM ============================================================

set PYTHONIOENCODING=utf-8
cd /d D:\carLane

echo Starting CarLaneI perception bench on http://127.0.0.1:8000 ...
echo (first model load takes ~15s; leave this window open)

REM Open the browser a few seconds after the server starts booting
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8000/"

D:\carLane\.venv\Scripts\python.exe -m uvicorn web.backend:app --host 127.0.0.1 --port 8000

pause
