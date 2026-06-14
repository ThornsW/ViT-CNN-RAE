# ViT-CNN-RAE — Claude Context

## 项目意图

基于 SRAE (TCSVT 2022) 的**注意力引导局部可逆对抗扰动**，跨 CNN 和 ViT 迁移。目标 CCF-C 期刊（计算机研究与发展 / PR Letters / Multimedia Tools and Applications / ICME / ICIP），有机会冲 TIFS short / IEEE TMM。

核心故事：SRAE 是全图扰动，浪费视觉质量预算；我们用 ViT 注意力定位 top-k 关键 patch，**只在小区域加扰动**，mask 位置作为可逆恢复的"密钥"。同等 ASR 下 PSNR 大幅提升。

## 用户背景（重要）

- 入门理论 + 能落地工程的研究新手
- 之前复现过 RAEncoder (CVPR 2024)，结果与原论文有差异
- **倾向用可信开源代码做基线，不愿用自己复现得不准的工作做对比** —— 不要推荐 RAEncoder 作为主基线
- 全局规则在 `~/.claude/CLAUDE.md`：Python 命令必须用绝对路径，禁止 base env，不确定环境时先问

## 文件布局（v0.4 — A 重度瘦身后的简洁版）

```
PROFILE                   # 单行 "research"
src/vit_cnn_rae/          # 可 import 的包,pip install -e .
├── config.py              # 路径 + env vars + run_dir() 内联
├── models/                # generator / discriminator / recover / blocks
├── attention/             # rollout / extractor / masking
├── attacks/               # base (SRAE 训练核 + save/load_checkpoint) + local
├── data/                  # dataset
├── targets/               # loader + trainers
├── evaluation/            # metrics + evaluator
├── visualization/         # save_images
├── report/                # Flask 结果可视化 web app(parse/charts/app + templates/static)
└── utils/                 # weights_init, plotting, seed (就这 3 个)

scripts/                  # 极简 CLI:argparse + 调 Attack/LocalAttack
├── train_baseline.py      # --seed --resume --target --epochs --batch-size
├── train_local.py         # 多 --top-k --attn-model
└── evaluate.py / save_visualizations.py / report_app.py / train_target_{densenet,resnet,mobilenet}.py / setup_autodl.sh

tests/test_smoke.py       # 2 个 case:attention 覆盖率 + LocalAttack end-to-end
data/splits/              # dataset-{trn,val}.txt
checkpoints/              # DenseNet121.pth (shipped)
outputs/<YYYYMMDD_HHMMSS_model_tag>/   # 时间戳隔离 (gitignored)
├── run_config.json       # 初始超参数 + git SHA + 开始时间 (训练启动时由 config.save_run_config 写)
└── models/{last.pth, epoch_NNN.pth, netG_epoch_NNN_1.pth, loss1.txt, *.png}
```

**v0.3 → v0.4 删了**:training/runner.py (200 行编排层)、utils/{tracking,checksums,logging_utils,run_id}.py、requirements-dev.txt、pytest.ini、sha256 sidecar、3 个调试旗标(--fast-dev-run / --overfit-one-batch / --debug / --tb / --seeds / --run-name)、3 个独立 smoke 文件。**保留**:模块化结构、config.py 集中路径、run_dir 时间戳隔离、--seed + set_seed、--resume + save/load_checkpoint、1 个 smoke 文件。**理由**:v2 全套对 1 人 research 项目过度;v0.4 ≈ 600 行可读代码。

## 关键设计决策

1. **不改 attacks/base.py 的 train_batch 训练逻辑**。新功能通过子类化 + 包装实现,baseline 可 bisect
2. `MaskedGenerator(nn.Module)` 包装 Generator,让 perturbation = inner(x) * mask
3. mask 由 `attention/extractor.py` 提供,从 `x`(clean)计算,作为密钥
4. timm >= 0.9 默认 `attn.fused_attn=True` 走 SDPA 跳过 attn_drop hook → `ViTAttentionExtractor` 强制设 False
5. 数据集路径列表里前缀是 `/caltech256/...`,`DATA_ROOT` 设为父目录(默认 `~/data`)
6. 所有路径走 `config.py` 不再硬编码,通过 env 覆盖:`DATA_ROOT` / `CHECKPOINT_DIR` / `OUTPUT_DIR`
7. **训练产物按时间戳目录隔离**:`outputs/<YYYYMMDD_HHMMSS_model_tag>/models/`,scripts 调 `config.run_dir(tag)` 生成
8. **极简至上**(v0.4):research 项目不需要 runner 编排层 / TensorBoard / sha256 / 6 个调试旗标。只有 `--seed` + `--resume` 真正需要

## 运行环境

| 角色 | 位置 | Python | GPU | 用途 |
|---|---|---|---|---|
| 本地开发 | `/home/thorns1exp/miniforge3/envs/DL/bin/python` | 3.12 | 无 | 代码 + report 可视化 web app |
| 云端训练 | AutoDL base env（按小时租用）| 默认 | 有 | 训练 150 epochs + 评估 |

> ⚠️ 本地 DL env **未装 torch**(只装了 report 用的 flask+matplotlib）。需要 torch 的 `tests/test_smoke.py` / 训练 / 评估只能在云端跑;本地仅 `tests/test_report.py`(torch-free)可跑。

```bash
# 安装
/home/thorns1exp/miniforge3/envs/DL/bin/pip install -e ".[dev]"   # 含 pytest

# 本地 smoke(2 个 case 必须过)
/home/thorns1exp/miniforge3/envs/DL/bin/python -m pytest tests/

# 云端训练
bash scripts/setup_autodl.sh
export DATA_ROOT=~/data
python scripts/train_baseline.py                                  # 单 seed=42, 150 ep
python scripts/train_local.py --top-k 0.2                         # 同
python scripts/evaluate.py --models-dir outputs/<run_dir>/models
python scripts/save_visualizations.py --models-dir outputs/<run_dir>/models

# 结果可视化 web app(本地浏览器看所有 run + 对比;torch-free,DL 装一次即可)
/home/thorns1exp/miniforge3/envs/DL/bin/pip install flask matplotlib       # report 只需这俩,不拉 torch
/home/thorns1exp/miniforge3/envs/DL/bin/python scripts/report_app.py --port 8000   # 开 http://127.0.0.1:8000

# 多 seed?手动跑几次(没必要写多 seed 编排)
for s in 42 123 2024; do python scripts/train_baseline.py --seed $s; done

# 实例被回收?resume
python scripts/train_local.py --resume outputs/<run_dir>/models/last.pth

# Git
# GitHub: https://github.com/ThornsW/ViT-CNN-RAE.git (私有)
# 本地无 SSH key / gh CLI,push 需在用户自己的终端做
```

## 当前状态（2026-05-31 结束）

完成：
- SRAE 基线 Linux/cloud 化
- attention + attack_local,CPU smoke test 通过(覆盖率 0.199)
- **v0.2 重构**:扁平目录 → src/vit_cnn_rae 包结构,路径全部走 config.py,scripts/ 集中 CLI 入口
- train_local.py 已写好(原 TODO 中的 main_local.py)
- 重构后 smoke test 重跑通过

**未推 GitHub**：用户需在自己终端跑 `git push -u origin main`(多个 commit 等推送)。

待办（按依赖顺序）：
1. 云端跑 SRAE baseline 150 epochs（`python scripts/train_baseline.py`），复现到 ASR ≥ 90% / 恢复 PSNR ≥ 45dB
2. 拿到真 Caltech 图，做注意力可视化验证（**Go/No-Go 决策点**：ViT 必须关注主物体而非背景，否则方案需要调整）
3. 跑 LocalAttack（`python scripts/train_local.py --top-k 0.2`）150 epochs
4. 对照实验：baseline vs LocalAttack，对比 ASR / PSNR / 恢复 PSNR
5. 消融：top_k_ratio (0.05/0.1/0.2/0.3)、attention 来源（单 ViT vs 多 ViT 集成 vs IG）
6. 鲁棒性：JPEG/blur/resize（社交平台场景）
7. 防御鲁棒性：AT/HGD/NRP/DiffPure/RS（用 TransferAttack 框架现成）
8. 跨架构迁移测试：5 CNN × 5 ViT 的 ASR 矩阵
9. 写论文

## 6 个月时间线

| 月 | 周 | 任务 | 状态 |
|---|---|---|---|
| M1 | W1-W2 | 环境 + 复现 baseline | 跳过（SRAE 直接拿到，用户已会 DL） |
| M1 | W3-W4 | 注意力提取 + RGAN 接入 | ✓ 完成（一天压缩） |
| M2 | W5-W8 | 跨模型 ASR 矩阵 + Go/No-Go | 待办 |
| M3 | W9-W12 | 局部 RGAN 联合训练 + 完整 pipeline | 待办 |
| M4 | W13-W16 | 主实验 + baseline 对比 + 方法图 | 待办 |
| M5 | W17-W20 | 消融 + 鲁棒性 + 防御鲁棒 | 待办 |
| M6 | W21-W24 | 写作 + 投稿 | 待办 |

## 已知坑

1. `timm` ViT 默认 fused_attn=True，attention 钩子失效。`ViTAttentionExtractor.__init__` 已处理
2. SRAE 的 save.py 上游 `models.Remover` 是 typo（应是 `Recover`），已修
3. SRAE evaluate.py 用 `skimage.measure.compare_*` 已废弃，已替换为 `skimage.metrics`
4. 本地 CPU `pin_memory=True` 会 warning，已改为 `pin_memory=torch.cuda.is_available()`
5. SRAE 上游 `os.environ["CUDA_VISIBLE_DEVICES"]="1"` 写死，AutoDL 的 GPU 是 index 0，已改 setdefault

## 投稿基线对比

主基线：**SRAE** (TCSVT 2022) — 直接对照对象，复现要严格
次基线：MIG (Ma 2023, ICCV)、TGR (Zhang 2023, ICCV)、FPR — 跨架构迁移攻击对比，用 TransferAttack 框架跑
**不用** RAEncoder 作基线（用户自己复现有差异）
