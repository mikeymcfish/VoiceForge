@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" >nul 2>&1
if errorlevel 1 exit /b 1

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Setup-SpeechRuntime.ps1" -Engine moss
set "SETUP_EXIT=%ERRORLEVEL%"

popd >nul 2>&1
exit /b %SETUP_EXIT%
