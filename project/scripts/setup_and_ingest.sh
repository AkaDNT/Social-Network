#!/usr/bin/env bash
set -euo pipefail

# 1) Check python & pip
python3 --version || python --version
pip --version

# 2) Install packages
pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt

# 3) Install pre-commit hooks
pre-commit install

# 4) Create data directories if missing
mkdir -p data/raw data/processed

echo "✅ Setup done."
echo "➡️  Put wiki-talk-temporal.txt.gz into data/raw/ and then run:"
echo "    python -m src.data.ingest --input data/raw/wiki-talk-temporal.txt.gz --out data/processed/wiki.parquet"

# Make this script executable (usually run once from terminal, not needed inside the script)
chmod +x scripts/setup_and_ingest.sh
