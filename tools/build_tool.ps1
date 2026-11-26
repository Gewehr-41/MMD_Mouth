<#
.SYNOPSIS
    Build the standalone rhythm-VMD tool (PyInstaller onedir) and zip it.

.DESCRIPTION
    Produces dist\节奏VMD工具\ containing:
        节奏VMD工具.exe   GUI entry
        ffmpeg.exe        audio decoding without Blender
        tools\ , mmd_mouth\   bundled python modules
    The 1.2 GB aligner model is NOT bundled: it is downloaded on first run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\build_tool.ps1
    powershell -ExecutionPolicy Bypass -File tools\build_tool.ps1 -Console
#>
param(
    [switch]$Console,
    [switch]$SkipBuild,
    [string]$OutName = "",
    [string]$EntryName = "MMDRhythmVMD"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# The display name lives in a UTF-8 file: Windows PowerShell 5.1 reads .ps1 as
# ANSI, so a Chinese literal here would turn into mojibake.
if (-not $OutName) {
    $nameFile = Join-Path $PSScriptRoot "tool_name.txt"
    if (Test-Path $nameFile) {
        $OutName = (Get-Content $nameFile -Encoding UTF8 | Select-Object -First 1).Trim()
    }
    if (-not $OutName) { $OutName = "RhythmVMD" }
}

$python = Join-Path $repo ".venv-worker\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "missing $python - create the worker venv first"
}

$distRoot = Join-Path $repo "build-tool\dist"
$workRoot = Join-Path $repo "build-tool\work"
$specRoot = Join-Path $repo "build-tool\spec"

if (-not $SkipBuild) {
    Write-Host "== PyInstaller ==" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir", "--noupx",
        "--name", $EntryName,
        "--distpath", $distRoot,
        "--workpath", $workRoot,
        "--specpath", $specRoot,
        "--paths", $repo,
        "--paths", (Join-Path $repo "tools"),
        # Only the pieces the tool needs: the add-on UI modules and the bundled
        # worker (mmd_mouth\runtime, ~250 MB) stay out of the standalone build.
        "--add-data", "$repo\mmd_mouth\__init__.py;mmd_mouth",
        "--add-data", "$repo\mmd_mouth\constants.py;mmd_mouth",
        "--add-data", "$repo\mmd_mouth\core;mmd_mouth\core",
        "--add-data", "$repo\mmd_mouth\language;mmd_mouth\language",
        "--add-data", "$repo\mmd_mouth\recognition;mmd_mouth\recognition",
        "--add-data", "$repo\tools;tools",
        "--collect-data", "pyopenjtalk",
        "--collect-data", "cmudict",
        "--collect-submodules", "pyopenjtalk",
        "--collect-binaries", "pyopenjtalk",
        "--copy-metadata", "cmudict",
        "--copy-metadata", "pyopenjtalk-plus",
        "--copy-metadata", "pypinyin",
        "--copy-metadata", "imageio-ffmpeg",
        "--hidden-import", "onnxruntime.capi._pybind_state",
        "--hidden-import", "cmudict",
        "--hidden-import", "pypinyin",
        "--hidden-import", "pyopenjtalk",
        "--hidden-import", "imageio_ffmpeg",
        "--exclude-module", "vosk",
        "--exclude-module", "matplotlib",
        "--exclude-module", "PySide6",
        "--exclude-module", "torch",
        "tool_entry.py"
    )
    if ($Console) { $pyArgs = $pyArgs[0..2] + @("--console") + $pyArgs[3..($pyArgs.Count - 1)] }
    else { $pyArgs = $pyArgs[0..2] + @("--windowed") + $pyArgs[3..($pyArgs.Count - 1)] }
    & $python @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with $LASTEXITCODE" }
}

$built = Join-Path $distRoot $EntryName
$target = Join-Path $distRoot $OutName
if (-not (Test-Path $built)) { throw "missing build output: $built" }

Write-Host "== ffmpeg ==" -ForegroundColor Cyan
$ffmpeg = Get-ChildItem -Path (Join-Path $repo ".venv-worker") -Recurse -Filter "ffmpeg*.exe" `
    -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $ffmpeg) {
    $ffmpeg = & $python -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())"
}
if (-not $ffmpeg -or -not (Test-Path $ffmpeg)) { throw "ffmpeg binary not found" }
Copy-Item $ffmpeg (Join-Path $built "ffmpeg.exe") -Force

Write-Host "== tidy ==" -ForegroundColor Cyan
Get-ChildItem $built -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
foreach ($script in @("rgba_to_wav.py", "render_mouth_sample.py", "analyze_keys.py")) {
    $candidate = Join-Path $built "tools\$script"
    if (Test-Path $candidate) { Remove-Item $candidate -Force }
}

if (Test-Path $target) { Remove-Item $target -Recurse -Force }
Move-Item $built $target

# PyInstaller needs an ASCII --name and the exe keeps that name: a Chinese
# exe name inside a Chinese folder failed to launch in testing, while either
# one being ASCII works.  The folder and the zip keep the friendly name.
$readme = Join-Path $PSScriptRoot "tool_readme.txt"
if (Test-Path $readme) { Copy-Item $readme (Join-Path $target "README.txt") -Force }

$size = (Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("folder: {0}  ({1:N1} MB)" -f $target, $size) -ForegroundColor Green

$zip = Join-Path $repo ("dist-addon\" + $OutName + ".zip")
New-Item -ItemType Directory -Force -Path (Split-Path $zip) | Out-Null
if (Test-Path $zip) { Remove-Item $zip -Force }
Write-Host "== zip ==" -ForegroundColor Cyan
Compress-Archive -Path $target -DestinationPath $zip -CompressionLevel Optimal
Write-Host ("zip: {0}  ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)) -ForegroundColor Green

Write-Host ""
Write-Host "Next: run '$target\$OutName.exe' - the model downloads on first launch." -ForegroundColor Yellow
