# ViT-CNN-RAE

Research repo: attention-guided **local** reversible adversarial perturbations that transfer across CNN and ViT.

Forked from the official PyTorch implementation of *Self-recoverable Adversarial Examples*
(Zhang et al., **TCSVT 2022**), which produces full-image RGAN perturbations. We replace the
global perturbation with one localized inside high-attention regions identified by a ViT,
and use the mask positions as an explicit recovery key.

The baseline (`main1.py`, `evaluate.py`, `save.py`, `RGAN.py`, `models.py`) is the original SRAE code
with Linux/cloud-friendly patches (paths configurable via `DATA_ROOT`, deprecated
`skimage.measure.compare_*` swapped for `skimage.metrics` equivalents, hardcoded GPU index
removed). All new modules will be added under separate files so the baseline stays bisectable.

## Setup

### Local (dev / smoke test, CPU OK)
```bash
pip install -U "torch>=2.0" "torchvision>=0.15" timm tqdm matplotlib pandas \
                "scikit-image>=0.19" scikit-learn pillow
```

### Cloud (AutoDL or similar, GPU)
```bash
bash scripts/setup_autodl.sh    # installs deps + downloads Caltech-256 to ~/data
export DATA_ROOT=~/data
python main1.py                  # 150 epochs RGAN training
```

## Data

[Caltech-256](https://authors.library.caltech.edu/7694/) (~1.2 GB, 30607 images).
Expected layout after extraction: `$DATA_ROOT/caltech256/256_ObjectCategories/<class>/<image>.jpg`.
Train/val splits are committed as `dataset-trn.txt` / `dataset-val.txt` (28037 / 2570).

## Pretrained target classifier

`DenseNet121.pth` (29 MB, 257-class Caltech-256 fine-tune) is shipped in the repo and used as the
target model for RGAN training. Optional ResNet50 / MobileNetV3 commented out in `main1.py`,
`evaluate.py`, `save.py` — uncomment if needed.

## Citing the original baseline

```bibtex
@ARTICLE{3207008,
  author={Zhang, Jiawei and Wang, Jinwei and Wang, Hao and Luo, Xiangyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks},
  year={2022},
  doi={10.1109/TCSVT.2022.3207008}}
```
