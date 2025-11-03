Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Install pre-commit the first time if needed (optional)
# pip install pre-commit

# Determine project root based on this .ps1 file location
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Create data folders
New-Item -ItemType Directory -Force -Path "data\raw"       | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null

# Install packages
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hook (optional)
pre-commit install

# Paths may contain spaces → always wrap in quotes
$Raw = Join-Path $ProjectRoot "data\raw\wiki-talk-temporal.txt.gz"
$Out = Join-Path $ProjectRoot "data\processed\wiki.parquet"

if (Test-Path "$Raw") {
    python -m src.data.ingest --input "$Raw" --out "$Out" --chunk_size 1000000
    Write-Host "✅ Ingest done: $Out"
} else {
    Write-Host "ℹ️  Raw file not found: $Raw"
    Write-Host "   Download the file, then run this command:"
    Write-Host "   python -m src.data.ingest --input data/raw/wiki-talk-temporal.txt.gz --out data/processed/wiki.parquet --chunk_size 1000000"
}
