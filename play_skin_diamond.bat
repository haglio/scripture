@echo off
REM Device-test launcher: MultiFunPlayer + VLC on the Sam Stone chapter.
REM Kills any stale VLC first (a background instance squats the HTTP control
REM port and MFP ends up talking to the wrong player -> "no funscript found").
REM MFP is reused if already running, started if not.
taskkill /F /IM vlc.exe >nul 2>&1
tasklist /FI "IMAGENAME eq MultiFunPlayer.exe" | find /I "MultiFunPlayer.exe" >nul || start "" "C:\Program Files\MultiFunPlayer-1.34.2-patreon\MultiFunPlayer.exe"
start "" "C:\Program Files\VideoLAN\VLC\vlc.exe" --no-one-instance --start-time=107 "C:\path\to\suite-root\videos\videos\2D\non_AI\winston\2 do not need work - originals are being or have been or will not be processed\redacted_540-redacted.mp4"
