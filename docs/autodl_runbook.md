# AutoDL 训练 Runbook

从零到 baseline + local 对比实验的完整命令清单。目标分类器 `checkpoints/DenseNet121.pth`
已随 git 仓库一起 clone(整库 ~31M），无需单独上传。

## ① 开机前（AutoDL 控制台）

- 租 GPU：RTX 3090 / 4090（24G）按量计费
- 镜像：PyTorch 2.x + CUDA 官方镜像
- 准备一个 GitHub Personal Access Token（私有仓库 clone 用）

## ② 上代码（二选一）

```bash
cd ~
# A. clone 私有仓库（<TOKEN> 换成你的 PAT）
git clone https://<TOKEN>@github.com/ThornsW/ViT-CNN-RAE.git
# B. 或用 AutoDL JupyterLab 直接上传整个项目文件夹
cd ViT-CNN-RAE
```

## ③ 装环境 + 下数据（一次性，约 10–20 分钟）

```bash
source /etc/network_turbo 2>/dev/null || true   # AutoDL 学术加速，没有就跳过
bash scripts/setup_autodl.sh
```

脚本动作：pip 装依赖 → wget 下载 256_ObjectCategories.tar（~1.2G）→ 解压到
`~/data/caltech256/256_ObjectCategories/` → 校验样例文件存在。

## ④ 后台跑 SRAE baseline（核心，数小时）

```bash
tmux new -s srae                                # 防 SSH 断线杀进程
export DATA_ROOT=~/data
python scripts/train_baseline.py 2>&1 | tee baseline.log
# 脱离会话：Ctrl-b 再按 d    重新接入：tmux attach -t srae
```

- 默认：densenet121 / 150 epoch / batch 22 / seed 42
- 显存不足：`--batch-size 16`（或更小）
- 产物目录：`outputs/<时间戳>_srae_baseline_s42/models/`
- ✅ 验收目标：ASR ≥ 90%、恢复 PSNR ≥ 45dB（对标 SRAE 原文）
- 中断恢复：`python scripts/train_baseline.py --resume outputs/<run>/models/last.pth`

## ⑤ 评估 + 可视化

```bash
export DATA_ROOT=~/data
RUN=$(ls -dt outputs/*_srae_baseline_* | head -1)
python scripts/evaluate.py --models-dir "$RUN/models"
python scripts/save_visualizations.py --models-dir "$RUN/models"
```

evaluate 打印 ASR + PSNR / SSIM / L0 / L2 / Linf。

## ⑥ 跑注意力引导的局部攻击 + 对比（创新点）

```bash
export DATA_ROOT=~/data HF_HUB_DISABLE_XET=1     # 这步要下 ViT 权重，必须加
python scripts/train_local.py --top-k 0.2 2>&1 | tee local.log
RUN_L=$(ls -dt outputs/*_srae_local_* | head -1)
python scripts/evaluate.py --models-dir "$RUN_L/models"
```

baseline vs local 的 ASR / PSNR 对比 = 论文核心数据。
消融可继续：`--top-k 0.05 / 0.1 / 0.3`。

## ⑦ 释放实例前务必取回结果

`outputs/` 不在 git 里，实例释放即丢失。用 JupyterLab 下载整个 `outputs/` 目录回本地
（权重 + loss 曲线 + 评估输出 + 可视化）。

## 四个必记的坑

1. 私有仓库 clone → 用 GitHub Token
2. 训练全程 tmux / nohup，否则 SSH 断线进程死
3. `train_local` 前 `export HF_HUB_DISABLE_XET=1`（本机/某些网络对 HF Xet 后端 SSL 必断）
4. 训练完先把 `outputs/` 下载回本地，再释放实例

## 数据下载备选（caltech.edu 在 AutoDL 上卡死时）

- 先 `source /etc/network_turbo` 开学术加速重试
- 仍不行：走 HF 镜像 `Qualeafclover/webdataset-caltech256`（+ `HF_HUB_DISABLE_XET=1`），
  内含 `.npy` 图 + `.json`（json 里 `original` 字段是原始 jpg 名，可据此重建
  `256_ObjectCategories/<NNN.class>/<name>.jpg` 结构以匹配 split）。需要时让 Claude 写重建脚本。
