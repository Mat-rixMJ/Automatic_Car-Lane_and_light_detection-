@echo off
REM ============================================================
REM  CarLaneI — One-Click Run
REM  Change the VIDEO path below, then double-click this file.
REM ============================================================

SET VIDEO=D:\carLane\downloads\frankfurt_720p_5min.mp4

REM ============================================================
REM  Options (uncomment what you need):
REM    --live          Show live window
REM    --no-record     Don't save output file
REM    --output "path" Save processed video to this path
REM    --no-crop       Don't auto-crop ultra-wide videos
REM ============================================================

D:\carLane\.venv\Scripts\python.exe D:\carLane\src\run_pipeline_fast.py --input "%VIDEO%" --output "D:\carLane\output\result.mp4" --live --no-crop

pause
