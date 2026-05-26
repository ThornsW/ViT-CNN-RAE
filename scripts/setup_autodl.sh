#!/usr/bin/env bash
# One-shot setup for AutoDL (or any Ubuntu cloud GPU machine).
# Pre-condition: base python with pip is already provided (AutoDL default).
# After this script: `export DATA_ROOT=~/data && python main1.py` to train.

set -euo pipefail

echo "[1/3] Installing Python deps into base env..."
pip install -U pip
pip install \
    "torch>=2.0" "torchvision>=0.15" \
    timm tqdm matplotlib pandas \
    "scikit-image>=0.19" scikit-learn pillow

echo "[2/3] Preparing Caltech-256 dataset under ~/data/caltech256/..."
DATA_DIR="${HOME}/data"
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
Setup complete. To start training:

    cd <repo_dir>
    export DATA_ROOT=${DATA_DIR}
    python main1.py

To run evaluation after training:

    python evaluate.py
================================================================
EOF
