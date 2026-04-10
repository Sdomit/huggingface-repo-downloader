param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipInstall) {
    python -m pip install --upgrade pip
    python -m pip install -e ".[build]"
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "HF Repo Downloader" `
    --distpath "dist" `
    --workpath "build/pyinstaller" `
    --specpath "build/pyinstaller" `
    --collect-submodules "keyring.backends" `
    --collect-submodules "keyring.util" `
    "hf_downloader/__main__.py"

Write-Host "Built dist/HF Repo Downloader/HF Repo Downloader.exe"
