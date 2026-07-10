@echo off
setlocal
cd /d "%~dp0"
title Nmotion

echo ============================================
echo   Nmotion - starting up
echo ============================================
echo.

REM --- Check Docker is installed ---
where docker >nul 2>&1
if errorlevel 1 (
  echo [X] Docker is not installed or not on PATH.
  echo     Install Docker Desktop from https://www.docker.com/products/docker-desktop
  echo.
  pause
  exit /b 1
)

REM --- Make sure Docker Desktop engine is running (start it if not) ---
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running yet. Trying to start it...
  start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  echo Waiting for Docker to be ready ^(this can take a minute^)...
  set /a tries=0
  :waitdocker
  timeout /t 3 >nul
  docker info >nul 2>&1
  if not errorlevel 1 goto dockerready
  set /a tries+=1
  if %tries% geq 40 (
    echo [X] Docker did not become ready. Open Docker Desktop manually, then run this again.
    pause
    exit /b 1
  )
  goto waitdocker
)
:dockerready
echo Docker is ready.
echo.

echo Building and starting Nmotion ^(first run downloads dependencies, be patient^)...
docker compose up --build -d
if errorlevel 1 (
  echo [X] Failed to start. See the messages above.
  pause
  exit /b 1
)

echo.
echo Waiting for the web app to come online...
set /a tries=0
:waitweb
timeout /t 2 >nul
curl -s -o nul http://localhost:8000/health
if not errorlevel 1 goto webready
set /a tries+=1
if %tries% geq 60 (
  echo App is taking a while. Opening the browser anyway - refresh if it's not up yet.
  goto openbrowser
)
goto waitweb
:webready
echo App is online.

:openbrowser
start "" http://localhost:8000
echo.
echo ============================================
echo   Nmotion is running at http://localhost:8000
echo   To stop it later, double-click stop.bat
echo ============================================
echo.
pause
