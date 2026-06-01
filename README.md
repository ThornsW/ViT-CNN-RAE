# ViT-CNN-RAE

> **PROFILE: research** — 论文复现项目

Attention-guided **local** reversible adversarial perturbations that transfer
across CNN and ViT. Forked from the official PyTorch implementation of
*Self-recoverable Adversarial Examples* (Zhang et al., **TCSVT 2022**), which
produces full-image RGAN perturbations. We replace the global perturbation
with one localised inside high-attention regions identified by a ViT, and use
the mask positions as an explicit recovery key.

## Reproduction alignment

| 项目 | 论文 (SRAE, TCSVT 2022) | 我们的复现 | 备注 |
|---|---|---|---|
| Dataset | Caltech-256 | Caltech-256 | — |
| Target classifier | DenseNet121 | DenseNet121 (257-class) | — |
| Generator error rate (ASR) | ≥ 90% | TBD | — |
| Recover PSNR | ≥ 45 dB | TBD | — |
| Epochs | 150 | 150 | — |
| Batch size | (unknown) | 22 | 单卡显存 |

填表时跑至少 2-3 个 seed,报 mean ± std。

## Verified environments

| Role | OS | Python | CUDA | GPU | 用途 | 日期 |
|---|---|---|---|---|---|---|
| dev | Linux | 3.12 (RAEncoder) | — | — | smoke test | 2026-05-31 |
| train | AutoDL Ubuntu | 3.10 | 12.1 | TBD | 150-ep 训练 | TBD |

## Layout

```
src/vit_cnn_rae/          # importable package (pip install -e .)
├── config.py              # paths + env vars + run_dir()
├── models/                # Generator, Discriminator, Recover, blocks
├── attention/             # ViT rollout, extractor, top-k masking
├── attacks/               # base (SRAE) + local (LocalAttack)
├── data/                  # MyDataset + default_transform
├── targets/               # load_target_model + train_classifier
├── evaluation/            # ASR + PSNR/SSIM/L0/L2/Linf
├── visualization/         # save_images
└── utils/                 # weights_init, set_seed, draw_loss_curve

scripts/                  # thin CLI shells
├── train_baseline.py      # SRAE baseline
├── train_local.py         # local variant, --top-k controls coverage
├── evaluate.py
├── save_visualizations.py
└── train_target_{densenet,resnet,mobilenet}.py

tests/test_smoke.py       # one pytest file, two cases
data/splits/              # train/val txt (committed)
checkpoints/              # DenseNet121.pth (shipped, 29 MB)
outputs/<YYYYMMDD_HHMMSS_model_tag>/   # per-run, gitignored
└── models/{last.pth, epoch_NNN.pth, netG_epoch_NNN_1.pth, loss1.txt, *.png}
```

## Setup

```bash
# install
pip install -e .                  # runtime
pip install -e ".[dev]"           # +pytest for smoke

# smoke (CPU OK)
pytest tests/

# cloud: dataset + 150-epoch training
bash scripts/setup_autodl.sh
export DATA_ROOT=~/data
python scripts/train_baseline.py
python scripts/train_local.py --top-k 0.2
```

## Training commands

```bash
# baseline (default seed=42)
python scripts/train_baseline.py
python scripts/train_baseline.py --seed 123    # different seed → different run dir

# local variant
python scripts/train_local.py --top-k 0.2
python scripts/train_local.py --top-k 0.1 --seed 123

# resume after instance preemption
python scripts/train_local.py --resume outputs/<run_dir>/models/last.pth
```

每次训练会在 `outputs/` 下生成一个时间戳目录,所有产物(权重 + 日志 + 曲线)都隔离在里面。

## Evaluation

```bash
# 把 outputs/<run_dir>/models 路径作为 --models-dir 传进去
python scripts/evaluate.py --models-dir outputs/<run_dir>/models
python scripts/save_visualizations.py --models-dir outputs/<run_dir>/models
```

## Env vars

| Variable          | Default                       | Purpose                          |
|-------------------|-------------------------------|----------------------------------|
| `DATA_ROOT`       | `~/data`                      | root of Caltech-256 images       |
| `CHECKPOINT_DIR`  | `<repo>/checkpoints`          | target classifier weights        |
| `OUTPUT_DIR`      | `<repo>/outputs`              | training products                |

## Data

[Caltech-256](https://authors.library.caltech.edu/7694/) (~1.2 GB).
Expected layout: `$DATA_ROOT/caltech256/256_ObjectCategories/<class>/<image>.jpg`.
Splits in `data/splits/dataset-{trn,val}.txt` (28037 / 2570).

## Citing the original baseline

```bibtex
@ARTICLE{3207008,
  author={Zhang, Jiawei and Wang, Jinwei and Wang, Hao and Luo, Xiangyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks},
  year={2022},
  doi={10.1109/TCSVT.2022.3207008}}
```
