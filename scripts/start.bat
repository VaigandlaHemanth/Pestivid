@echo off
REM ============================================================================
REM  PestVid - the only launcher.
REM
REM  There used to be nine .bat files starting five different combinations of
REM  servers across ports 3000/3001/3002/5000/8080, and the README documented a
REM  sixth. This is the single canonical entry point.
REM
REM  PORT MAP (the only one):
REM    3001  Node/Express API  +  serves public/ as static files
REM    5000  Flask AI server   (optional - disease detection and RAG chat)
REM
REM  The frontend is served BY the API on 3001, so window.__PESTVID resolves to
REM  same-origin /api and /ml. Open http://127.0.0.1:3001 - do NOT open
REM  public/index.html off disk, because file:// breaks CORS on every request.
REM ============================================================================
setlocal
cd /d "%~dp0"
title PestVid

echo.
echo  ============================================
echo    PestVid
echo  ============================================
echo.

REM ---- Node ----
where node >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found. Install from https://nodejs.org
    goto :fail
)

REM ---- config ----
if not exist "backend\.env" (
    echo  [ERROR] backend\.env is missing.
    echo          Copy backend\.env.example to backend\.env and fill it in.
    echo          MONGODB_URI and JWT_SECRET are required - the server now
    echo          refuses to start without them instead of half-booting.
    goto :fail
)

REM ---- deps ----
if not exist "backend\node_modules" (
    echo  [SETUP] Installing backend dependencies...
    pushd backend && call npm install && popd
)

REM ---- API + frontend on 3001 ----
echo  [1/2] Starting Node API + frontend on http://127.0.0.1:3001 ...
start "PestVid API" cmd /k "cd /d %~dp0backend && npm start"
timeout /t 4 /nobreak >nul

REM ---- Flask AI on 5000 (optional) ----
where python >nul 2>&1
if errorlevel 1 (
    echo  [2/2] Python not found - skipping the AI server.
    echo        Disease detection and RAG chat will report 503, by design.
) else (
    echo  [2/2] Starting Flask AI server on http://127.0.0.1:5000 ...
    start "PestVid AI" cmd /k "cd /d %~dp0 && python flask_server.py"
)

timeout /t 2 /nobreak >nul
echo.
echo  ============================================
echo    Open:  http://127.0.0.1:3001
echo  ============================================
echo.
echo   Seeded demo accounts (from backend\seed.js):
echo     farmer    demo.farmer@pestivid.sim   / password123
echo     buyer     demo.buyer@pestivid.sim    / password123
echo     investor  demo.investor@pestivid.sim / password123
echo.
echo   No MongoDB installed? Run instead:  cd backend ^&^& npm run dev:mem
echo   (starts an in-memory MongoDB and seeds it automatically)
echo.
start "" "http://127.0.0.1:3001"
goto :end

:fail
echo.
echo  Startup aborted.
:end
echo  Press any key to close this launcher...
pause >nul
endlocal
