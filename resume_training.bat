@echo off
REM ============================================================
REM  Resume ego-lane training from the last checkpoint.
REM  Picks up exactly where it stopped (epoch, optimizer, LR).
REM  When done, also run finalize_overnight.py to lock in results.
REM ============================================================

set PYTHONIOENCODING=utf-8

D:\carLane\.venv\Scripts\python.exe D:\carLane\src\train_ego_seg.py --resume

echo.
echo Training resumed-and-finished. Now finalizing (engine + eval + demo)...
D:\carLane\.venv\Scripts\python.exe D:\carLane\src\finalize_overnight.py
pause
