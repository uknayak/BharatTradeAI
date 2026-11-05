@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "FRONTEND=%ROOT%frontend"

if not exist "%PYTHON%" (
    echo Creating the project Python environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 -m venv "%ROOT%.venv"
    ) else (
        python -m venv "%ROOT%.venv"
    )
    if errorlevel 1 goto :failed
    "%PYTHON%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 goto :failed
)

where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js and npm are required but were not found in PATH.
    goto :failed
)

if not exist "%FRONTEND%\node_modules\.bin\ng.cmd" (
    echo Installing frontend dependencies...
    pushd "%FRONTEND%"
    call npm install
    if errorlevel 1 (
        popd
        goto :failed
    )
    popd
)

netstat -ano -p tcp | findstr /R /C:":8000 .*LISTENING" >nul
if errorlevel 1 (
    echo Starting the FastAPI backend...
    start "BharatTradeAI Backend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "$root = '%ROOT%'; Set-Location -LiteralPath $root; $env:DATABASE_URL = 'sqlite:///./bharattradeai.db'; & '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
) else (
    echo Backend is already listening on port 8000; leaving it running.
)

netstat -ano -p tcp | findstr /R /C:":4200 .*LISTENING" >nul
if errorlevel 1 (
    echo Starting the Angular frontend...
    start "BharatTradeAI Frontend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%FRONTEND%'; npm start"
) else (
    echo Frontend is already listening on port 4200; leaving it running.
)

echo.
echo Frontend: http://127.0.0.1:4200/
echo Backend:  http://127.0.0.1:8000/docs
echo Health:   http://127.0.0.1:8000/api/v1/health
timeout /t 5 /nobreak >nul
exit /b 0

:failed
echo.
echo Startup could not continue. Review the message above and install missing prerequisites.
pause
exit /b 1