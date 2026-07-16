@echo off
setlocal EnableExtensions DisableDelayedExpansion

pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
  echo [VoiceForge] Could not open the application directory.
  exit /b 1
)

if /I "%~1"=="help" goto show_help
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
if not "%~1"=="" (
  echo [VoiceForge] Unknown IndexTTS setup option: %~1
  goto show_help_error
)

echo.
echo [VoiceForge] Preparing the pinned official IndexTTS runtime...
echo [VoiceForge] This one-time setup downloads Python packages and may take a while.
echo [VoiceForge] Existing model weights will not be downloaded again.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Setup-IndexTTS.ps1"
set "SETUP_EXIT=%ERRORLEVEL%"

if not "%SETUP_EXIT%"=="0" (
  echo.
  echo [VoiceForge] IndexTTS setup did not complete.
  if /I not "%VOICEFORGE_NO_PAUSE%"=="1" pause
  popd >nul 2>&1
  exit /b %SETUP_EXIT%
)

echo.
echo [VoiceForge] IndexTTS runtime setup is complete.
echo [VoiceForge] Start VoiceForge again, then click Verify runtime.
popd >nul 2>&1
exit /b 0

:show_help
echo.
echo VoiceForge IndexTTS runtime setup
echo.
echo Usage:
echo   Setup-IndexTTS.cmd
echo   VoiceForge.cmd setup-index
echo.
echo The setup creates an isolated Python 3.11 environment from the pinned
echo official index-tts/index-tts source under attached_assets\index-tts\runtime.
echo It preserves existing model downloads and writes only the two IndexTTS
echo runtime paths to the ignored .env file.
popd >nul 2>&1
exit /b 0

:show_help_error
echo Run "Setup-IndexTTS.cmd help" for usage.
if /I not "%VOICEFORGE_NO_PAUSE%"=="1" pause
popd >nul 2>&1
exit /b 1
