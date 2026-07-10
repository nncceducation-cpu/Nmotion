@echo off
cd /d "%~dp0"
title Nmotion - stop
echo Stopping Nmotion...
docker compose down
echo Done. You can close this window.
pause
