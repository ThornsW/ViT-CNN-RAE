#!/usr/bin/env bash
# One-shot setup for AutoDL (or any Ubuntu cloud GPU machine).
# Pre-condition: base python with pip is already provided (AutoDL default).
# After this script: `python scripts/train_baseline.py` to train.
# Data lives in <repo>/data, which is config.py's default DATA_ROOT (no export needed).

set -euo pipefail

echo "[1/3] Installing the vit_cnn_rae package (editable) + deps..."
pip install -U pip
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
pip install -e "${REPO_DIR}"

echo "[2/3] Preparing Caltech-256 dataset under ${REPO_DIR}/data/caltech256/..."
DATA_DIR="${REPO_DIR}/data"
mkdir -p "${DATA_DIR}/caltech256"
cd "${DATA_DIR}"

if [ ! -d "caltech256/256_ObjectCategories" ]; then
    if [ ! -f "256_ObjectCategories.tar" ]; then
        echo "  -> Downloading 256_ObjectCategories.tar (~1.2GB)..."
        wget -c "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
    fi
    echo "  -> Extracting (this takes a couple minutes)..."
    tar -xf 256_ObjectCategories.tar -C caltech256/
else
    echo "  -> Caltech-256 already present, skipping."
fi

echo "[3/3] Sanity-check expected paths..."
SAMPLE="${DATA_DIR}/caltech256/256_ObjectCategories/001.ak47/001_0042.jpg"
if [ ! -f "${SAMPLE}" ]; then
    echo "  !! Expected file not found: ${SAMPLE}"
    echo "  !! Check that the tarball extracted correctly."
    exit 1
fi
echo "  -> OK. Sample file exists: ${SAMPLE}"

cat <<EOF

================================================================
Setup complete. DATA_ROOT defaults to ${DATA_DIR} (no export needed).
To start training:

    cd ${REPO_DIR}
    python scripts/train_baseline.py             # SRAE baseline
    python scripts/train_local.py --top-k 0.2    # attention-guided local

To run evaluation after training:

    python scripts/evaluate.py
    python scripts/save_visualizations.py
================================================================
EOF
