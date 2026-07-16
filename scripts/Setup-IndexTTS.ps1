[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$OfficialRepository = "https://github.com/index-tts/index-tts.git"
$PinnedRevision = "13495845e3028f0bb6ca1462ad22aa0e76349e40"
$Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$RuntimeRoot = [IO.Path]::GetFullPath((Join-Path $Workspace "attached_assets\index-tts\runtime"))
$SourceDirectory = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot "source"))
$PythonExecutable = [IO.Path]::GetFullPath((Join-Path $SourceDirectory ".venv\Scripts\python.exe"))
$RuntimeManifest = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot "voiceforge-index-runtime.json"))
$DotEnvPath = [IO.Path]::GetFullPath((Join-Path $Workspace ".env"))

function Write-Step([string] $Message) {
  Write-Host "[VoiceForge] $Message"
}

function Get-RequiredCommand([string] $Name, [string] $InstallHint) {
  $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $command) {
    throw "$Name is required. $InstallHint"
  }
  return $command.Source
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

function Get-ExternalOutput {
  param(
    [Parameter(Mandatory = $true)][string] $FilePath,
    [Parameter(Mandatory = $true)][string[]] $Arguments,
    [string] $WorkingDirectory = $Workspace
  )

  Push-Location -LiteralPath $WorkingDirectory
  $previousErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorAction
    Pop-Location
  }

  if ($exitCode -ne 0) {
    throw "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
  }
  return (($output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
}

function Test-OfficialOrigin([string] $Origin) {
  $normalized = $Origin.Trim().TrimEnd("/").ToLowerInvariant()
  return $normalized -in @(
    "https://github.com/index-tts/index-tts.git",
    "https://github.com/index-tts/index-tts",
    "git@github.com:index-tts/index-tts.git"
  )
}

function Assert-ManagedPath([string] $Candidate) {
  $runtimePrefix = $RuntimeRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
  if (-not $Candidate.StartsWith($runtimePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to manage a path outside the VoiceForge IndexTTS runtime directory: $Candidate"
  }
}

function Set-IndexEnvironment([string] $PythonPath, [string] $SourcePath) {
  $keys = @("INDEX_TTS_PYTHON", "INDEX_TTS_SOURCE_DIR")
  $lines = if (Test-Path -LiteralPath $DotEnvPath) {
    [Collections.Generic.List[string]]::new([IO.File]::ReadAllLines($DotEnvPath))
  } else {
    [Collections.Generic.List[string]]::new()
  }

  $filtered = [Collections.Generic.List[string]]::new()
  foreach ($line in $lines) {
    $isManaged = $false
    if ($line.Trim() -eq "# Managed by Setup-IndexTTS.cmd") {
      $isManaged = $true
    }
    foreach ($key in $keys) {
      if ($line -match ("^\s*(?:export\s+)?" + [regex]::Escape($key) + "\s*=")) {
        $isManaged = $true
        break
      }
    }
    if (-not $isManaged) {
      $filtered.Add($line)
    }
  }

  while ($filtered.Count -gt 0 -and [string]::IsNullOrWhiteSpace($filtered[$filtered.Count - 1])) {
    $filtered.RemoveAt($filtered.Count - 1)
  }
  if ($filtered.Count -gt 0) {
    $filtered.Add("")
  }
  $filtered.Add("# Managed by Setup-IndexTTS.cmd")
  $filtered.Add(('INDEX_TTS_PYTHON="{0}"' -f $PythonPath.Replace("\", "/")))
  $filtered.Add(('INDEX_TTS_SOURCE_DIR="{0}"' -f $SourcePath.Replace("\", "/")))

  $temporaryPath = "$DotEnvPath.voiceforge-index.tmp"
  $encoding = [Text.UTF8Encoding]::new($false)
  try {
    [IO.File]::WriteAllText(
      $temporaryPath,
      (($filtered -join [Environment]::NewLine) + [Environment]::NewLine),
      $encoding
    )
    Move-Item -LiteralPath $temporaryPath -Destination $DotEnvPath -Force
  } finally {
    if (Test-Path -LiteralPath $temporaryPath) {
      Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
  }
}

try {
  Assert-ManagedPath $SourceDirectory
  Assert-ManagedPath $PythonExecutable
  Assert-ManagedPath $RuntimeManifest

  $git = Get-RequiredCommand "git.exe" "Install Git for Windows from https://git-scm.com/download/win."
  $uv = Get-RequiredCommand "uv.exe" "Install uv from https://docs.astral.sh/uv/getting-started/installation/."
  Invoke-External -FilePath $git -Arguments @("lfs", "version")

  New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
  $env:UV_CACHE_DIR = Join-Path $RuntimeRoot "uv-cache"
  $env:UV_LINK_MODE = "copy"
  $env:UV_PROJECT_ENVIRONMENT = Join-Path $SourceDirectory ".venv"
  $freshClone = $false

  if (-not (Test-Path -LiteralPath $SourceDirectory)) {
    Write-Step "Cloning the pinned official IndexTTS source..."
    $previousSmudge = $env:GIT_LFS_SKIP_SMUDGE
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    try {
      Invoke-External -FilePath $git -Arguments @(
        "-c", "core.longpaths=true", "clone", "--filter=blob:none", "--no-checkout",
        $OfficialRepository, $SourceDirectory
      )
      $freshClone = $true
    } finally {
      if ($null -eq $previousSmudge) {
        Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
      } else {
        $env:GIT_LFS_SKIP_SMUDGE = $previousSmudge
      }
    }
  } elseif (-not (Test-Path -LiteralPath (Join-Path $SourceDirectory ".git"))) {
    $entries = @(Get-ChildItem -LiteralPath $SourceDirectory -Force -ErrorAction SilentlyContinue)
    if ($entries.Count -gt 0) {
      throw "The managed source directory is non-empty but is not a Git checkout: $SourceDirectory"
    }
    throw "Remove the empty managed source directory and run setup again: $SourceDirectory"
  }

  $origin = Get-ExternalOutput -FilePath $git -Arguments @("remote", "get-url", "origin") -WorkingDirectory $SourceDirectory
  if (-not (Test-OfficialOrigin $origin)) {
    throw "Refusing unexpected IndexTTS origin '$origin'. Expected $OfficialRepository."
  }

  if (-not $freshClone) {
    $worktreeStatus = Get-ExternalOutput -FilePath $git -Arguments @("status", "--porcelain", "--untracked-files=all") -WorkingDirectory $SourceDirectory
    if ($worktreeStatus) {
      throw "The managed IndexTTS checkout has local changes. Preserve or remove them manually before setup continues."
    }
  }

  Write-Step "Fetching reviewed source revision $($PinnedRevision.Substring(0, 12))..."
  Invoke-External -FilePath $git -Arguments @("fetch", "--depth", "1", "origin", $PinnedRevision) -WorkingDirectory $SourceDirectory
  Invoke-External -FilePath $git -Arguments @("checkout", "--detach", $PinnedRevision) -WorkingDirectory $SourceDirectory
  $head = Get-ExternalOutput -FilePath $git -Arguments @("rev-parse", "HEAD") -WorkingDirectory $SourceDirectory
  if ($head.ToLowerInvariant() -ne $PinnedRevision) {
    throw "The official checkout did not resolve to the reviewed revision."
  }
  $checkedOutStatus = Get-ExternalOutput -FilePath $git -Arguments @("status", "--porcelain", "--untracked-files=all") -WorkingDirectory $SourceDirectory
  if ($checkedOutStatus) {
    throw "The pinned IndexTTS checkout is not clean after checkout; refusing to continue."
  }

  Invoke-External -FilePath $git -Arguments @("lfs", "install", "--local") -WorkingDirectory $SourceDirectory
  Invoke-External -FilePath $git -Arguments @("lfs", "pull") -WorkingDirectory $SourceDirectory

  Write-Step "Creating the official locked Python 3.11 environment with uv..."
  Invoke-External -FilePath $uv -Arguments @("sync", "--frozen") -WorkingDirectory $SourceDirectory
  if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "uv completed but the expected IndexTTS Python executable is missing: $PythonExecutable"
  }

  Write-Step "Validating Python, PyTorch, and the official IndexTTS import..."
  $previousPythonPath = $env:PYTHONPATH
  $previousSource = $env:VOICEFORGE_INDEX_SOURCE
  $env:PYTHONPATH = $SourceDirectory
  $env:VOICEFORGE_INDEX_SOURCE = $SourceDirectory
  $validationCode = @'
import json
import inspect
import os
import pathlib
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise RuntimeError(f"IndexTTS requires Python 3.10 or 3.11; found {sys.version.split()[0]}")

import torch

parts = torch.__version__.split("+", 1)[0].split(".")
if tuple(int(part) for part in parts[:2]) < (2, 6):
    raise RuntimeError(f"VoiceForge requires PyTorch 2.6 or newer; found {torch.__version__}")

import indextts.infer_v2 as infer_v2

source = pathlib.Path(os.environ["VOICEFORGE_INDEX_SOURCE"]).resolve()
module_path = pathlib.Path(infer_v2.__file__).resolve()
module_path.relative_to(source)
if not hasattr(infer_v2, "IndexTTS2"):
    raise RuntimeError("The pinned official source does not expose IndexTTS2")

constructor_parameters = set(inspect.signature(infer_v2.IndexTTS2).parameters)
required_constructor = {"cfg_path", "model_dir", "use_fp16", "device", "use_cuda_kernel", "use_deepspeed"}
missing_constructor = required_constructor - constructor_parameters
if missing_constructor:
    raise RuntimeError(f"Pinned IndexTTS2 constructor is missing expected parameters: {sorted(missing_constructor)}")

infer_parameters = set(inspect.signature(infer_v2.IndexTTS2.infer).parameters)
required_infer = {"spk_audio_prompt", "text", "output_path", "verbose"}
missing_infer = required_infer - infer_parameters
if missing_infer:
    raise RuntimeError(f"Pinned IndexTTS2 inference API is missing expected parameters: {sorted(missing_infer)}")

print("VOICEFORGE_RUNTIME_JSON=" + json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "module": str(module_path)}))
'@
  # Windows PowerShell 5.1 rewrites embedded quotes when a multiline string is
  # passed to a native executable with `python -c`. Write the validator to a
  # managed temporary file so Python receives its contents byte-for-byte.
  $validationScript = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot ".voiceforge-index-validation.py"))
  Assert-ManagedPath $validationScript
  [IO.File]::WriteAllText($validationScript, $validationCode, [Text.UTF8Encoding]::new($false))
  $previousErrorAction = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $validationLines = @(& $PythonExecutable $validationScript 2>&1 | ForEach-Object { $_.ToString() })
    $validationExit = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorAction
    Remove-Item -LiteralPath $validationScript -Force -ErrorAction SilentlyContinue
    if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    if ($null -eq $previousSource) { Remove-Item Env:VOICEFORGE_INDEX_SOURCE -ErrorAction SilentlyContinue } else { $env:VOICEFORGE_INDEX_SOURCE = $previousSource }
  }
  if ($validationExit -ne 0) {
    throw "IndexTTS runtime validation failed:`n$($validationLines -join [Environment]::NewLine)"
  }
  $validationPrefix = "VOICEFORGE_RUNTIME_JSON="
  $validationLine = $validationLines | Where-Object { $_.StartsWith($validationPrefix) } | Select-Object -Last 1
  if (-not $validationLine) {
    throw "IndexTTS runtime validation produced no machine-readable result."
  }
  $validation = $validationLine.Substring($validationPrefix.Length) | ConvertFrom-Json

  $manifest = [ordered]@{
    schema_version = 1
    repository = $OfficialRepository
    revision = $PinnedRevision
    source_directory = $SourceDirectory.Replace("\", "/")
    python_executable = $PythonExecutable.Replace("\", "/")
    python_version = $validation.python
    torch_version = $validation.torch
    configured_at = [DateTimeOffset]::Now.ToString("o")
  }
  $manifestTemporary = "$RuntimeManifest.tmp"
  $manifestJson = $manifest | ConvertTo-Json -Depth 4
  [IO.File]::WriteAllText($manifestTemporary, "$manifestJson`n", [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $manifestTemporary -Destination $RuntimeManifest -Force

  Set-IndexEnvironment -PythonPath $PythonExecutable -SourcePath $SourceDirectory

  Write-Step "Validated official IndexTTS at $($PinnedRevision.Substring(0, 12))."
  Write-Step "Saved the isolated runtime paths to the ignored .env file."
  Write-Step "Restart VoiceForge before selecting Verify runtime."
  exit 0
} catch {
  Write-Host "[VoiceForge] ERROR: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
