# Install packages into the worker venv without pip's index round-trips.
#
# pip stalls while fetching package indexes on this machine, but the files
# themselves download instantly.  This script asks the mirror's JSON API for
# the newest release that still supports Python 3.10, then either extracts the
# wheel (they are zip files) or the sdist's package folder into site-packages.

param(
    [string]$Python = "",
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn",
    [string]$Tag = "cp310",
    [string[]]$Packages
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $repo ".venv-worker\Scripts\python.exe" }
$wheelDir = Join-Path $repo "wheels"
$sitePackages = Join-Path $repo ".venv-worker\Lib\site-packages"
New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null
$ProgressPreference = "SilentlyContinue"

function Test-Python27([string]$requires) {
    if (-not $requires) { return $true }
    if ($requires -match ">=\s*3\.(1[1-9]|[2-9][0-9])") { return $false }
    if ($requires -match ">\s*3\.10") { return $false }
    return $true
}

function Select-File($release) {
    $files = @($release)
    # Binary wheels: either built for this interpreter (cp310) or stable-ABI
    # wheels (cp39-abi3), which fit any CPython >= 3.9.
    $wheel = $files | Where-Object {
        $_.filename -match "\.whl$" -and $_.filename -match "win_amd64" -and
        ($_.filename -match $Tag -or $_.filename -match "abi3")
    } | Select-Object -First 1
    if (-not $wheel) { $wheel = $files | Where-Object { $_.filename -match "(py3|py2\.py3)-none-win_amd64\.whl$" } | Select-Object -First 1 }
    if (-not $wheel) { $wheel = $files | Where-Object { $_.filename -match "(py3|py2\.py3)-none-any\.whl$" } | Select-Object -First 1 }
    if ($wheel) { return $wheel }
    return $files | Where-Object { $_.filename -match "\.tar\.gz$" } | Select-Object -First 1
}

$results = @()
$Packages = $Packages | ForEach-Object { $_ -split "," } | Where-Object { $_ }
foreach ($package in $Packages) {
    try {
        $meta = Invoke-RestMethod "$Mirror/pypi/$package/json" -TimeoutSec 90
    }
    catch {
        $results += "$package : METADATA FAILED"
        continue
    }

    $versions = @($meta.releases.PSObject.Properties.Name)
    $ordered = $versions | Sort-Object { try { [version]($_ -replace "[^0-9.]", "") } catch { [version]"0.0.0" } } -Descending

    $file = $null
    $chosen = $null
    foreach ($version in $ordered) {
        if (-not (Test-Python27 $meta.releases.$version.requires_python)) { continue }
        $candidate = Select-File $meta.releases.$version
        if ($candidate) { $file = $candidate; $chosen = $version; break }
    }
    if (-not $file) { $results += "$package : NO COMPATIBLE FILE"; continue }

    $target = Join-Path $wheelDir $file.filename
    try {
        Invoke-WebRequest -Uri $file.url -OutFile $target -UseBasicParsing -TimeoutSec 300
        if ($file.filename -match "\.whl$") {
            & $Python -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $target $sitePackages
        }
        else {
            & $Python -c "import sys,tarfile,pathlib; t=tarfile.open(sys.argv[1]); t.extractall(sys.argv[2])" $target $wheelDir
            $root = Get-ChildItem $wheelDir -Directory | Where-Object { $_.Name -like ($file.filename -replace "\.tar\.gz$", "") } | Select-Object -First 1
            if ($root) {
                Get-ChildItem $root.FullName -Directory | Where-Object { $_.Name -notin @("tests", "test", "docs") } | ForEach-Object {
                    Copy-Item $_.FullName -Destination $sitePackages -Recurse -Force
                }
            }
        }
        $results += "$package : OK $chosen ($($file.filename))"
    }
    catch {
        $results += "$package : FAILED $($_.Exception.Message)"
    }
}
$results | ForEach-Object { Write-Host $_ }
