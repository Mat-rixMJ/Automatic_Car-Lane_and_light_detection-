@echo off
REM ============================================================
REM  CarLaneI - One-Click LIVE Demo (no frontend needed)
REM
REM  Runs the full perception stack (trained ego-lane + sign
REM  detector + traffic-light state + vehicles) LIVE in a window
REM  on one of the three Indian showcase clips.
REM
REM  This is the fallback if the web UI isn't available - just
REM  double-click and pick a clip.
REM ============================================================

REM Delayed expansion so filenames with ( ) & ' don't break parsing
setlocal EnableDelayedExpansion

REM Force UTF-8 so console prints never crash on this machine
set PYTHONIOENCODING=utf-8

set ROOT=D:\carLane
set PY=%ROOT%\.venv\Scripts\python.exe
set PIPE=%ROOT%\src\run_pipeline_fast.py

:menu
cls
echo ============================================================
echo   CarLaneI - LIVE Perception Demo (Indian roads)
echo ============================================================
echo.
echo   Select a FULL-LENGTH clip to run live (long validation):
echo.
echo     [1]  Kolkata      - dense city traffic          (~45 min)
echo     [2]  Mumbai       - busy metro roads            (~29 min)
echo     [3]  India Night  - NH-44 highway, low light    (~3.5 min)
echo.
echo     [Q]  Quit
echo.
set /p CHOICE=  Enter choice (1/2/3/Q): 

REM Resolve the chosen full-length source video (labeled sections keep the
REM special-character filenames safe — no fragile & chaining).
if /I "%CHOICE%"=="1" goto pick1
if /I "%CHOICE%"=="2" goto pick2
if /I "%CHOICE%"=="3" goto pick3
if /I "%CHOICE%"=="Q" goto end

echo.
echo   Invalid choice. Try again.
timeout /t 2 >nul
goto menu

:pick1
set NAME=Kolkata
set "VIDEO=%ROOT%\vidssave.com 4K Drive in Kolkata _ East India's Tier-1 City 720P.mp4"
goto run

:pick2
set NAME=Mumbai
set "VIDEO=%ROOT%\vidssave.com 4K Drive to South Mumbai, India (via Mahim) 720P.mp4"
goto run

:pick3
set NAME=India Night
set "VIDEO=%ROOT%\india_night_full_fixed.mp4"
goto run

:run
echo.
if not exist "!VIDEO!" (
    echo   ERROR: clip not found:
    echo     !VIDEO!
    echo.
    pause
    goto menu
)
echo   Running LIVE on: !NAME!  (full-length, fast cadence ~23 FPS)
echo   A window will open with the annotated feed.
echo   Press  Q  or  ESC  in that window to stop early.
echo.
echo   (live-only - no file written, so the long run stays smooth)
echo.

"%PY%" "%PIPE%" --input "!VIDEO!" --no-crop --no-record --live --display-h 720 --fast

echo.
echo   Stopped / finished: !NAME!
echo.
set /p AGAIN=  Run another clip? (Y = menu, any other key = exit): 
if /I "!AGAIN!"=="Y" goto menu
goto end

:end
endlocal
