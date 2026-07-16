@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem VoiceForge Studio Windows launcher.
rem Double-click for a production launch, or run "VoiceForge.cmd help" for options.

pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
  echo [VoiceForge] Could not open the application directory:
  echo   %~dp0
  call :pause_if_needed
  exit /b 1
)

set "MODE=production"
set "OPEN_BROWSER=1"
set "FORCE_INSTALL=0"
set "PORT_OVERRIDE="
set "APP_PORT="
set "APP_URL="
set "DEPS_HASH="
set "DEPS_STAMP=node_modules\.voiceforge-deps.sha256"
set "NODE_INSTALL_ATTEMPTED=0"
if not defined NPM_CONFIG_CACHE set "NPM_CONFIG_CACHE=%CD%\.npm-cache"

:parse_arguments
if "%~1"=="" goto arguments_ready

if /I "%~1"=="run" (
  set "MODE=production"
  shift
  goto parse_arguments
)
if /I "%~1"=="start" (
  set "MODE=production"
  shift
  goto parse_arguments
)
if /I "%~1"=="production" (
  set "MODE=production"
  shift
  goto parse_arguments
)
if /I "%~1"=="prod" (
  set "MODE=production"
  shift
  goto parse_arguments
)
if /I "%~1"=="dev" (
  set "MODE=development"
  shift
  goto parse_arguments
)
if /I "%~1"=="install" (
  set "MODE=install"
  shift
  goto parse_arguments
)
if /I "%~1"=="repair" (
  set "MODE=install"
  set "FORCE_INSTALL=1"
  shift
  goto parse_arguments
)
if /I "%~1"=="setup-index" (
  set "MODE=setup-index"
  shift
  goto parse_arguments
)
if /I "%~1"=="setup-qwen" (
  set "MODE=setup-qwen"
  shift
  goto parse_arguments
)
if /I "%~1"=="setup-moss" (
  set "MODE=setup-moss"
  shift
  goto parse_arguments
)
if /I "%~1"=="--reinstall" (
  set "FORCE_INSTALL=1"
  shift
  goto parse_arguments
)
if /I "%~1"=="--no-browser" (
  set "OPEN_BROWSER=0"
  shift
  goto parse_arguments
)
if /I "%~1"=="--port" (
  if "%~2"=="" goto missing_port
  set "PORT_OVERRIDE=%~2"
  shift
  shift
  goto parse_arguments
)
if /I "%~1"=="help" goto show_help
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
if /I "%~1"=="/?" goto show_help

echo [VoiceForge] Unknown option: %~1
echo.
goto show_help_error

:missing_port
echo [VoiceForge] --port requires a number from 1 through 65535.
goto fail

:arguments_ready
if not exist "package.json" (
  echo [VoiceForge] package.json is missing from:
  echo   %CD%
  goto fail
)
if not exist "package-lock.json" (
  echo [VoiceForge] package-lock.json is required for a deterministic install.
  goto fail
)

if /I "%MODE%"=="setup-index" goto run_index_setup
if /I "%MODE%"=="setup-qwen" goto run_qwen_setup
if /I "%MODE%"=="setup-moss" goto run_moss_setup
goto continue_standard_startup

:run_index_setup
call "%~dp0Setup-IndexTTS.cmd"
set "INDEX_SETUP_EXIT=%ERRORLEVEL%"
popd >nul 2>&1
exit /b %INDEX_SETUP_EXIT%

:run_qwen_setup
call "%~dp0Setup-QwenTTS.cmd"
set "QWEN_SETUP_EXIT=%ERRORLEVEL%"
popd >nul 2>&1
exit /b %QWEN_SETUP_EXIT%

:run_moss_setup
call "%~dp0Setup-MossTTS.cmd"
set "MOSS_SETUP_EXIT=%ERRORLEVEL%"
popd >nul 2>&1
exit /b %MOSS_SETUP_EXIT%

:continue_standard_startup

call :ensure_node
if errorlevel 1 goto fail

call :ensure_npm
if errorlevel 1 goto fail

rem A repeat launch must discover the already-running app before npm ci. Native
rem Vite/Rolldown modules stay locked by a live Node process on Windows.
if /I "%MODE%"=="install" goto dependency_check

if defined PORT_OVERRIDE set "PORT=%PORT_OVERRIDE%"
call :resolve_port
if errorlevel 1 goto fail

set "APP_URL=http://127.0.0.1:%APP_PORT%/"
call :inspect_port
set "PORT_STATE=%ERRORLEVEL%"
if "%PORT_STATE%"=="0" goto already_running
if "%PORT_STATE%"=="2" goto preferred_port_in_use
goto dependency_check

:preferred_port_in_use
if defined PORT_OVERRIDE goto port_in_use
set "PREFERRED_PORT=%APP_PORT%"
call :find_free_port
if errorlevel 1 goto port_in_use
set "PORT=%APP_PORT%"
set "APP_URL=http://127.0.0.1:%APP_PORT%/"
echo.
echo [VoiceForge] Port %PREFERRED_PORT% is busy; using available port %APP_PORT% instead.

:dependency_check
call :calculate_dependency_hash
if errorlevel 1 goto fail

call :ensure_dependencies
if errorlevel 1 goto fail

if /I "%MODE%"=="install" (
  echo.
  echo [VoiceForge] Installation is ready.
  goto success
)

:port_ready

if /I "%MODE%"=="production" (
  echo.
  echo [VoiceForge] Building the production app...
  call npm.cmd run build
  if errorlevel 1 (
    echo.
    echo [VoiceForge] The production build did not complete.
    goto fail
  )
)

echo.
echo [VoiceForge] Starting at %APP_URL%
echo [VoiceForge] Press Ctrl+C in this window to stop the server.

if "%OPEN_BROWSER%"=="1" call :start_browser_waiter

if /I "%MODE%"=="development" (
  call npm.cmd run dev
) else (
  call npm.cmd start
)
set "SERVER_EXIT=%ERRORLEVEL%"

echo.
echo [VoiceForge] Server stopped with exit code %SERVER_EXIT%.
if not "%SERVER_EXIT%"=="0" call :pause_if_needed
popd >nul 2>&1
exit /b %SERVER_EXIT%

:already_running
echo.
echo [VoiceForge] VoiceForge is already running at %APP_URL%
if "%OPEN_BROWSER%"=="1" start "" "%APP_URL%"
goto success

:port_in_use
echo.
echo [VoiceForge] Port %APP_PORT% is already in use by another application.
echo [VoiceForge] Stop that application or choose another port, for example:
echo   VoiceForge.cmd --port 5050
goto fail

:ensure_node
set "NODE_REASON=missing"
where node.exe >nul 2>&1
if errorlevel 1 goto offer_node_install

node -e "const v=process.versions.node.split('.').map(Number);process.exit(v[0]>22||(v[0]===22&&v[1]>=12)?0:1)" >nul 2>&1
if errorlevel 1 (
  set "NODE_REASON=outdated"
  goto offer_node_install
)

for /f "delims=" %%V in ('node -p "process.versions.node"') do set "NODE_VERSION=%%V"
echo [VoiceForge] Node.js %NODE_VERSION%
exit /b 0

:offer_node_install
echo.
if "%NODE_INSTALL_ATTEMPTED%"=="1" (
  echo [VoiceForge] Node.js was updated, but this shell still sees the old version.
  echo [VoiceForge] Close this window and launch VoiceForge.cmd again.
  exit /b 1
)
if /I not "%NODE_REASON%"=="outdated" goto node_missing_message
for /f "delims=" %%V in ('node -p "process.versions.node" 2^>nul') do set "NODE_VERSION=%%V"
echo [VoiceForge] Node.js %NODE_VERSION% is too old. Version 22.12 or newer is required.
goto node_install_choice

:node_missing_message
echo [VoiceForge] Node.js 22.12 or newer is required but was not found.

:node_install_choice

where winget.exe >nul 2>&1
if errorlevel 1 goto manual_node_install

choice /C YN /N /M "Install the current Node.js LTS with winget now? [Y/N] "
if errorlevel 2 goto manual_node_install

echo.
if /I "%NODE_REASON%"=="outdated" (
  winget upgrade --id OpenJS.NodeJS.LTS --exact --accept-source-agreements --accept-package-agreements
) else (
  winget install --id OpenJS.NodeJS.LTS --exact --accept-source-agreements --accept-package-agreements
)
if errorlevel 1 (
  echo [VoiceForge] winget could not complete the Node.js installation.
  goto manual_node_install
)

set "NODE_INSTALL_ATTEMPTED=1"
set "PATH=%ProgramFiles%\nodejs;%PATH%"
where node.exe >nul 2>&1
if errorlevel 1 (
  echo.
  echo [VoiceForge] Node.js was installed. Close this window and launch VoiceForge.cmd again.
  exit /b 1
)
goto ensure_node

:manual_node_install
echo.
echo [VoiceForge] Install the current Node.js LTS, then run this launcher again:
echo   https://nodejs.org/en/download
exit /b 1

:ensure_npm
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [VoiceForge] npm.cmd was not found. Repair or reinstall Node.js LTS.
  exit /b 1
)

set "NPM_MAJOR="
set "NPM_VERSION="
for /f "tokens=1 delims=." %%V in ('npm.cmd --version 2^>nul') do set "NPM_MAJOR=%%V"
for /f "delims=" %%V in ('npm.cmd --version 2^>nul') do set "NPM_VERSION=%%V"
if not defined NPM_MAJOR (
  echo [VoiceForge] npm did not report a version.
  exit /b 1
)
if %NPM_MAJOR% LSS 10 (
  echo [VoiceForge] npm %NPM_VERSION% is too old. npm 10 or newer is required.
  echo [VoiceForge] Update Node.js LTS and try again.
  exit /b 1
)
echo [VoiceForge] npm %NPM_VERSION%
exit /b 0

:calculate_dependency_hash
for /f "delims=" %%H in ('node -e "const c=require('crypto'),f=require('fs');console.log(c.createHash('sha256').update(f.readFileSync('package.json')).update(f.readFileSync('package-lock.json')).digest('hex'))" 2^>nul') do set "DEPS_HASH=%%H"
if not defined DEPS_HASH (
  echo [VoiceForge] Could not calculate the package-file checksum.
  exit /b 1
)
exit /b 0

:ensure_dependencies
set "NEED_INSTALL=1"
set "INSTALLED_HASH="
if exist "%DEPS_STAMP%" set /p "INSTALLED_HASH=" < "%DEPS_STAMP%"

if "%FORCE_INSTALL%"=="0" if /I "%INSTALLED_HASH%"=="%DEPS_HASH%" if exist "node_modules\.bin\tsx.cmd" if exist "node_modules\.bin\vite.cmd" set "NEED_INSTALL=0"

if "%NEED_INSTALL%"=="0" call npm.cmd ls --depth=0 --include=dev --silent >nul 2>&1
if "%NEED_INSTALL%"=="0" if errorlevel 1 set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="0" (
  echo [VoiceForge] JavaScript dependencies are current.
  exit /b 0
)

echo.
if "%FORCE_INSTALL%"=="1" (
  echo [VoiceForge] Repairing the locked JavaScript dependencies...
) else (
  echo [VoiceForge] Installing the locked JavaScript dependencies...
)

call :report_dependency_lockers
if errorlevel 1 (
  echo [VoiceForge] Dependency replacement cannot run while those processes are active.
  echo [VoiceForge] Stop VoiceForge or close its server terminal, then rerun this command.
  exit /b 1
)

call npm.cmd ci --include=dev --no-audit --no-fund
set "NPM_CI_EXIT=%ERRORLEVEL%"
if "%NPM_CI_EXIT%"=="0" goto npm_ci_completed

call :report_dependency_lockers
if errorlevel 1 goto npm_ci_locked
if "%NPM_CI_EXIT%"=="-4048" goto npm_ci_retry
if "%NPM_CI_EXIT%"=="-4051" goto npm_ci_retry
goto npm_ci_failed

:npm_ci_retry
echo.
echo [VoiceForge] Windows temporarily blocked dependency cleanup; retrying once...
powershell.exe -NoLogo -NoProfile -NonInteractive -Command "Start-Sleep -Milliseconds 1500" >nul 2>&1
call npm.cmd ci --include=dev --no-audit --no-fund
set "NPM_CI_EXIT=%ERRORLEVEL%"
if "%NPM_CI_EXIT%"=="0" goto npm_ci_completed
call :report_dependency_lockers
if errorlevel 1 goto npm_ci_locked
goto npm_ci_failed

:npm_ci_locked
echo.
echo [VoiceForge] npm ci failed with exit code %NPM_CI_EXIT%.
echo [VoiceForge] A running Node process is using files under node_modules.
echo [VoiceForge] Stop the listed VoiceForge process or close its terminal, then rerun this command.
exit /b 1

:npm_ci_failed
echo.
echo [VoiceForge] npm ci failed with exit code %NPM_CI_EXIT%.
echo [VoiceForge] Check the network, proxy, antivirus, permissions, and available disk space.
exit /b 1

:npm_ci_completed

if not exist "node_modules\.bin\tsx.cmd" (
  echo [VoiceForge] npm reported success, but the tsx launcher is missing.
  exit /b 1
)
if not exist "node_modules\.bin\vite.cmd" (
  echo [VoiceForge] npm reported success, but the Vite launcher is missing.
  exit /b 1
)

call npm.cmd ls --depth=0 --include=dev --silent >nul 2>&1
set "NPM_LS_EXIT=%ERRORLEVEL%"
if not "%NPM_LS_EXIT%"=="0" (
  echo [VoiceForge] The installed dependency tree did not pass npm validation.
  exit /b 1
)

> "%DEPS_STAMP%" echo %DEPS_HASH%
echo [VoiceForge] Dependencies installed successfully.
exit /b 0

:resolve_port
set "APP_PORT="
for /f "delims=" %%P in ('node --env-file-if-exists=.env -p "process.env.PORT || '5000'" 2^>nul') do set "APP_PORT=%%P"
set "VOICEFORGE_CANDIDATE_PORT=%APP_PORT%"
powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$parsed=0; $raw=$env:VOICEFORGE_CANDIDATE_PORT; if($raw -match '^[0-9]+$' -and [int]::TryParse($raw,[ref]$parsed) -and $parsed -ge 1 -and $parsed -le 65535){exit 0}; exit 1" >nul 2>&1
if errorlevel 1 (
  set "VOICEFORGE_CANDIDATE_PORT="
  set "APP_PORT="
  echo [VoiceForge] PORT must be a whole number from 1 through 65535.
  exit /b 1
)
set "APP_PORT=%VOICEFORGE_CANDIDATE_PORT%"
set "VOICEFORGE_CANDIDATE_PORT="
exit /b 0

:report_dependency_lockers
set "VOICEFORGE_LOCKING_NODE_PIDS="
set "VOICEFORGE_PROJECT_ROOT=%CD%"
for /f "delims=" %%P in ('powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$root=[IO.Path]::GetFullPath((Join-Path $env:VOICEFORGE_PROJECT_ROOT 'node_modules')); $ids=[Collections.Generic.List[int]]::new(); foreach($process in (Get-Process node -ErrorAction SilentlyContinue)){try{foreach($module in $process.Modules){if($module.FileName -and $module.FileName.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)){if(-not $ids.Contains($process.Id)){$ids.Add($process.Id)}; break}}}catch{}}; if($ids.Count -gt 0){[Console]::Write([string]::Join(', ',[int[]]$ids))}" 2^>nul') do set "VOICEFORGE_LOCKING_NODE_PIDS=%%P"
set "VOICEFORGE_PROJECT_ROOT="
if not defined VOICEFORGE_LOCKING_NODE_PIDS exit /b 0
echo [VoiceForge] Node process ID(s) holding project dependencies: %VOICEFORGE_LOCKING_NODE_PIDS%
exit /b 1

:inspect_port
powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$url='%APP_URL%'; try { $response=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1; if ($response.Content -match 'VoiceForge Studio') { exit 0 } } catch {}; $client=[Net.Sockets.TcpClient]::new(); try { $task=$client.ConnectAsync('127.0.0.1',%APP_PORT%); if ($task.Wait(500) -and $client.Connected) { exit 2 } } catch {} finally { $client.Dispose() }; exit 1" >nul 2>&1
exit /b %ERRORLEVEL%

:find_free_port
set "FREE_PORT="
for /f "delims=" %%P in ('powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$start=%APP_PORT%; $last=[Math]::Min(65535,$start+20); for ($candidate=$start+1; $candidate -le $last; $candidate++) { $listener=$null; try { $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$candidate); $listener.Start(); $listener.Stop(); Write-Output $candidate; exit 0 } catch { if ($null -ne $listener) { try { $listener.Stop() } catch {} } } }; exit 1" 2^>nul') do set "FREE_PORT=%%P"
if not defined FREE_PORT exit /b 1
set "APP_PORT=%FREE_PORT%"
exit /b 0

:start_browser_waiter
start "" /b powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -Command "$url='%APP_URL%'; for ($i=0; $i -lt 180; $i++) { try { $response=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1; if ($response.Content -match 'VoiceForge Studio') { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }" >nul 2>&1
exit /b 0

:show_help
echo.
echo VoiceForge Studio Windows launcher
echo.
echo Usage:
echo   VoiceForge.cmd                 Smart install, build, start, and open the app
echo   VoiceForge.cmd dev             Start the development server
echo   VoiceForge.cmd install         Install dependencies if package files changed
echo   VoiceForge.cmd repair          Force a clean npm ci and exit
echo   VoiceForge.cmd setup-index     Install the isolated official IndexTTS runtime
echo   VoiceForge.cmd setup-qwen      Install the isolated Qwen3-TTS runtime
echo   VoiceForge.cmd setup-moss      Install the isolated MOSS-TTS v1.5 runtime
echo   VoiceForge.cmd production      Build and start production mode
echo.
echo Options:
echo   --reinstall                    Force npm ci before starting
echo   --no-browser                   Do not open the browser automatically
echo   --port NUMBER                  Override PORT for this launch
echo   help                           Show this help
echo.
echo Speech models and their Python environments remain explicit setup steps
echo inside VoiceForge; normal startup never downloads model weights.
goto success

:show_help_error
echo Run "VoiceForge.cmd help" to see the supported modes and options.
goto fail

:success
popd >nul 2>&1
exit /b 0

:fail
call :pause_if_needed
popd >nul 2>&1
exit /b 1

:pause_if_needed
if /I "%VOICEFORGE_NO_PAUSE%"=="1" exit /b 0
echo.
pause
exit /b 0
