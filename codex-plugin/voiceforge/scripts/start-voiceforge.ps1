[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$VoiceForgeRoot,

  [Parameter(Mandatory = $true)]
  [ValidateRange(1, 65535)]
  [int]$Port,

  [Parameter(Mandatory = $true)]
  [string]$LauncherLogPath,

  [Parameter(Mandatory = $true)]
  [string]$ProcessIdPath
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($VoiceForgeRoot)
$launcher = Join-Path $root "VoiceForge.cmd"
$package = Join-Path $root "package.json"
if (-not [IO.Path]::IsPathRooted($root) -or -not (Test-Path -LiteralPath $launcher -PathType Leaf) -or -not (Test-Path -LiteralPath $package -PathType Leaf)) {
  throw "VoiceForgeRoot is not a valid VoiceForge installation."
}

Set-Location -LiteralPath $root
$stdoutLog = "$LauncherLogPath.stdout.log"
$stderrLog = "$LauncherLogPath.stderr.log"
$process = Start-Process `
  -FilePath $launcher `
  -ArgumentList @("--no-browser", "--port", [string]$Port) `
  -WorkingDirectory $root `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru
if ($null -eq $process) { throw "VoiceForge could not be launched." }
[IO.File]::WriteAllText([IO.Path]::GetFullPath($ProcessIdPath), [string]$process.Id)
exit 0
