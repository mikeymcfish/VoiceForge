[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("qwen", "moss")]
  [string] $Engine
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$EngineConfig = if ($Engine -eq "qwen") {
  [ordered]@{
    Label = "Qwen3-TTS"
    Root = [IO.Path]::GetFullPath((Join-Path $Workspace "attached_assets\qwen3-tts"))
    EnvKey = "QWEN_TTS_PYTHON"
    FfmpegBinKey = $null
    Packages = @(
      "torch==2.8.0",
      "torchaudio==2.8.0",
      "qwen-tts==0.1.1"
    )
    Validation = @'
from importlib import metadata
import torch
import torchaudio
import qwen_tts

assert metadata.version("qwen-tts") == "0.1.1"
assert tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) >= (2, 8)
print(f"Qwen3-TTS runtime ready: torch={torch.__version__}, qwen-tts={metadata.version('qwen-tts')}")
'@
  }
} else {
  [ordered]@{
    Label = "MOSS-TTS v1.5"
    Root = [IO.Path]::GetFullPath((Join-Path $Workspace "attached_assets\moss-tts-v1.5"))
    EnvKey = "MOSS_TTS_PYTHON"
    FfmpegBinKey = "MOSS_TTS_FFMPEG_BIN"
    Packages = @(
      "torch==2.9.1",
      "torchaudio==2.9.1",
      "torchcodec==0.8.1",
      "transformers==5.0.0",
      "safetensors==0.6.2",
      "numpy==2.1.0",
      "orjson==3.11.4",
      "tqdm==4.67.1",
      "PyYAML==6.0.3",
      "einops==0.8.1",
      "scipy==1.16.2",
      "librosa==0.11.0",
      "tiktoken==0.12.0",
      "soundfile==0.13.1",
      "huggingface_hub"
    )
    Validation = @'
from importlib import metadata
import os

_ffmpeg_dll_directory = None
if os.name == "nt":
    ffmpeg_bin = os.environ.get("MOSS_TTS_FFMPEG_BIN", "").strip()
    if not ffmpeg_bin:
        raise RuntimeError("MOSS_TTS_FFMPEG_BIN is not configured")
    _ffmpeg_dll_directory = os.add_dll_directory(ffmpeg_bin)

import torch
import torchaudio
import torchcodec
import transformers

assert metadata.version("transformers") == "5.0.0"
assert tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) >= (2, 9)
print(f"MOSS-TTS runtime ready: torch={torch.__version__}, transformers={transformers.__version__}")
'@
  }
}

$RuntimeRoot = [IO.Path]::GetFullPath((Join-Path $EngineConfig.Root "runtime"))
$VenvRoot = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot ".venv"))
$PythonExecutable = [IO.Path]::GetFullPath((Join-Path $VenvRoot "Scripts\python.exe"))
$DotEnvPath = [IO.Path]::GetFullPath((Join-Path $Workspace ".env"))

function Write-Step([string] $Message) {
  Write-Host "[VoiceForge] $Message"
}

function Get-RequiredCommand([string] $Name, [string] $Hint) {
  $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $command) { throw "$Name is required. $Hint" }
  return $command.Source
}

function Assert-ManagedPath([string] $Candidate) {
  $prefix = $RuntimeRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
  if ($Candidate -ne $RuntimeRoot -and -not $Candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to manage a path outside the ${Engine} runtime: $Candidate"
  }
}

function Invoke-External {
  param(
    [Parameter(Mandatory = $true)][string] $FilePath,
    [Parameter(Mandatory = $true)][string[]] $Arguments,
    [string] $WorkingDirectory = $Workspace
  )
  Push-Location -LiteralPath $WorkingDirectory
  try {
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
  } finally {
    Pop-Location
  }
  if ($exitCode -ne 0) {
    throw "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')"
  }
}

function Set-EnvironmentValue([string] $Key, [string] $Value) {
  $lines = if (Test-Path -LiteralPath $DotEnvPath) {
    [Collections.Generic.List[string]]::new([IO.File]::ReadAllLines($DotEnvPath))
  } else {
    [Collections.Generic.List[string]]::new()
  }
  $output = [Collections.Generic.List[string]]::new()
  foreach ($line in $lines) {
    if ($line -match ("^\s*(?:export\s+)?" + [regex]::Escape($Key) + "\s*=")) { continue }
    if ($line.Trim() -eq "# Managed by VoiceForge speech runtime setup: $Engine") { continue }
    $output.Add($line)
  }
  while ($output.Count -gt 0 -and [string]::IsNullOrWhiteSpace($output[$output.Count - 1])) {
    $output.RemoveAt($output.Count - 1)
  }
  if ($output.Count -gt 0) { $output.Add("") }
  $output.Add("# Managed by VoiceForge speech runtime setup: $Engine")
  $output.Add(('{0}="{1}"' -f $Key, $Value.Replace("\", "/")))
  $temporary = "$DotEnvPath.voiceforge-speech.tmp"
  try {
    [IO.File]::WriteAllText(
      $temporary,
      (($output -join [Environment]::NewLine) + [Environment]::NewLine),
      [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $DotEnvPath -Force
  } finally {
    if (Test-Path -LiteralPath $temporary) {
      Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
  }
}

function Test-CompatibleSharedFfmpegBin([string] $Candidate) {
  if ([string]::IsNullOrWhiteSpace($Candidate)) { return $false }
  try {
    $resolved = [IO.Path]::GetFullPath($Candidate.Trim().Trim('"'))
  } catch {
    return $false
  }
  if (-not (Test-Path -LiteralPath $resolved -PathType Container)) { return $false }
  if (-not (Test-Path -LiteralPath (Join-Path $resolved "ffmpeg.exe") -PathType Leaf)) { return $false }

  $compatibleSets = @(
    @("avcodec-58.dll", "avfilter-7.dll", "avformat-58.dll", "avutil-56.dll", "swresample-3.dll", "swscale-5.dll"),
    @("avcodec-59.dll", "avfilter-8.dll", "avformat-59.dll", "avutil-57.dll", "swresample-4.dll", "swscale-6.dll"),
    @("avcodec-60.dll", "avfilter-9.dll", "avformat-60.dll", "avutil-58.dll", "swresample-4.dll", "swscale-7.dll"),
    @("avcodec-61.dll", "avfilter-10.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll", "swscale-8.dll")
  )
  foreach ($requiredNames in $compatibleSets) {
    $complete = $true
    foreach ($name in $requiredNames) {
      if (-not (Test-Path -LiteralPath (Join-Path $resolved $name) -PathType Leaf)) {
        $complete = $false
        break
      }
    }
    if ($complete) { return $true }
  }
  return $false
}

function Find-CompatibleSharedFfmpegBin {
  $candidates = [Collections.Generic.List[string]]::new()
  if (-not [string]::IsNullOrWhiteSpace($env:MOSS_TTS_FFMPEG_BIN)) {
    $candidates.Add($env:MOSS_TTS_FFMPEG_BIN)
  }
  if (Test-Path -LiteralPath $DotEnvPath -PathType Leaf) {
    foreach ($line in [IO.File]::ReadAllLines($DotEnvPath)) {
      if ($line -match '^\s*MOSS_TTS_FFMPEG_BIN\s*=\s*"([^"]+)"\s*$') {
        $candidates.Add($Matches[1])
      }
    }
  }

  $ffmpegCommand = Get-Command "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -ne $ffmpegCommand) {
    $candidates.Add((Split-Path -Parent $ffmpegCommand.Source))
  }
  foreach ($entry in (($env:Path -split [IO.Path]::PathSeparator) | Where-Object { $_ })) {
    $candidates.Add($entry)
  }

  $packageRoots = @(
    (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"),
    (Join-Path $env:ProgramFiles "WinGet\Packages")
  )
  foreach ($packageRoot in $packageRoots) {
    if (-not (Test-Path -LiteralPath $packageRoot -PathType Container)) { continue }
    foreach ($package in (Get-ChildItem -LiteralPath $packageRoot -Directory -Filter "Gyan.FFmpeg.Shared*" -ErrorAction SilentlyContinue)) {
      foreach ($distribution in (Get-ChildItem -LiteralPath $package.FullName -Directory -ErrorAction SilentlyContinue)) {
        $candidates.Add((Join-Path $distribution.FullName "bin"))
      }
    }
  }

  $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  foreach ($candidate in $candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
    try {
      $resolved = [IO.Path]::GetFullPath($candidate.Trim().Trim('"'))
    } catch {
      continue
    }
    if (-not $seen.Add($resolved)) { continue }
    if (Test-CompatibleSharedFfmpegBin $resolved) { return $resolved }
  }
  return $null
}

try {
  Assert-ManagedPath $VenvRoot
  Assert-ManagedPath $PythonExecutable
  $uv = Get-RequiredCommand "uv.exe" "Install uv from https://docs.astral.sh/uv/getting-started/installation/."
  $sharedFfmpegBin = $null
  if ($Engine -eq "moss") {
    $sharedFfmpegBin = Find-CompatibleSharedFfmpegBin
    if ([string]::IsNullOrWhiteSpace($sharedFfmpegBin)) {
      $detected = Get-Command "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
      $detectedText = if ($null -ne $detected) {
        " Found ffmpeg.exe at $($detected.Source), but it is not a compatible shared build."
      } else {
        ""
      }
      throw ("MOSS-TTS requires FFmpeg 4-7 with shared DLLs for TorchCodec on Windows." +
        $detectedText + [Environment]::NewLine +
        "Install the compatible user-scoped build with:" + [Environment]::NewLine +
        "  winget install --id Gyan.FFmpeg.Shared --exact --version 7.1.1 --scope user" +
        [Environment]::NewLine +
        "Then rerun VoiceForge.cmd setup-moss. For a custom install, set MOSS_TTS_FFMPEG_BIN to its bin directory.")
    }
    $env:MOSS_TTS_FFMPEG_BIN = $sharedFfmpegBin
    $env:Path = "$sharedFfmpegBin$([IO.Path]::PathSeparator)$env:Path"
    Write-Step "Using compatible shared FFmpeg from $sharedFfmpegBin."
  } else {
    Get-RequiredCommand "ffmpeg.exe" "Install FFmpeg and add it to PATH before using local speech models." | Out-Null
  }

  New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
  $env:UV_CACHE_DIR = Join-Path $RuntimeRoot "uv-cache"
  $env:UV_LINK_MODE = "copy"

  if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    Write-Step "Creating an isolated Python 3.12 environment for $($EngineConfig.Label)..."
    Invoke-External -FilePath $uv -Arguments @("venv", "--python", "3.12", $VenvRoot)
  } else {
    Write-Step "Reusing the isolated environment at $VenvRoot."
  }

  Write-Step "Installing the pinned $($EngineConfig.Label) runtime..."
  $installArguments = @("pip", "install", "--python", $PythonExecutable, "--torch-backend", "auto")
  $installArguments += [string[]]$EngineConfig.Packages
  Invoke-External -FilePath $uv -Arguments $installArguments

  Write-Step "Validating imports and pinned versions..."
  $validationPath = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot ".voiceforge-runtime-check.py"))
  Assert-ManagedPath $validationPath
  try {
    [IO.File]::WriteAllText($validationPath, $EngineConfig.Validation, [Text.UTF8Encoding]::new($false))
    Invoke-External -FilePath $PythonExecutable -Arguments @($validationPath)
  } finally {
    Remove-Item -LiteralPath $validationPath -Force -ErrorAction SilentlyContinue
  }

  if ($Engine -eq "moss") {
    Set-EnvironmentValue -Key $EngineConfig.FfmpegBinKey -Value $sharedFfmpegBin
  }
  Set-EnvironmentValue -Key $EngineConfig.EnvKey -Value $PythonExecutable
  Write-Step "$($EngineConfig.Label) runtime is ready."
  Write-Step "Restart VoiceForge, open Create audio, and download the pinned model snapshot."
  exit 0
} catch {
  Write-Host "[VoiceForge] ERROR: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
