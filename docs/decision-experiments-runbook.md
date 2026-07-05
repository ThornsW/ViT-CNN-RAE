# 裁决实验云端操作单(B 部分)

> 目的:用最小 GPU 成本跑出 3 个判决数字,决定论文走路线 A(注意力局部+密钥)还是路线 B(SSL 特征攻击)。
> 云端用 base env,命令直接用 `python`(本机 DL env 无 torch,只能开发不能跑训练/评估)。

## B0 环境同步(先在本地 `git push`,云端再 pull)

```bash
cd <项目路径>            # AutoDL 上项目所在目录
git pull
bash scripts/setup_autodl.sh
pip install lpips        # A5 新依赖(timm 已在 setup 里)
```

## B4 训练 victim(必须在 B5 之前)

```bash
python scripts/train_target_resnet.py
python scripts/train_target_mobilenet.py
python scripts/train_target_vit.py --model vit_base_patch16_224      # 同源 ViT
python scripts/train_target_vit.py --model deit3_small_patch16_224   # 异源 ViT(关键:非自攻击)
```

权重存到 `checkpoints/{ResNet50,MoblieNetV3,ViT_B16,DeiT3_S16}.pth`。

## B1 零训练评估包(现有 5 个 run,<1h)

```bash
# 门控恢复 + LPIPS,配置自动从 run_config 读(baseline 会自动忽略 --gated-recovery)
for R in outputs/*_srae_baseline_s42 outputs/*_srae_local_*; do
  python scripts/evaluate.py --models-dir "$R/models" --gated-recovery
done

# mask 重合率(只对 local run)
for R in outputs/*_srae_local_*; do
  python scripts/mask_overlap.py --models-dir "$R/models"
done
```

出:恢复 PSNR 的密钥门控/自恢复两列、LPIPS 表、重合率(带随机下界)。

## B2 resume 硬 mask+eps32 到 150ep(~2.5h)

```bash
python scripts/train_local.py \
  --resume outputs/20260622_111910_srae_local_local_topk0_2_bg0_0_eps0_125_s42/models/last.pth \
  --epochs 150
python scripts/evaluate.py \
  --models-dir outputs/20260622_111910_srae_local_local_topk0_2_bg0_0_eps0_125_s42/models --gated-recovery
```

出:硬 mask 的 ASR 天花板(现 91%,精调阶段后还能不能涨)。

## B3 baseline+eps32 对照(~6h,审稿人必问的解耦实验)

```bash
python scripts/train_baseline.py --eps 0.125
python scripts/evaluate.py --models-dir outputs/*_srae_baseline_eps0.125_s42/models
```

出:全图扰动 + 同样 eps 约束的基准,用来剥离"局部化"与"eps 约束"各自的贡献。

## B5 迁移矩阵(~1h,路线 A vs B 的判决)

```bash
python scripts/transfer_matrix.py \
  --runs outputs/*_srae_baseline_s42 \
         outputs/*_srae_local_local_topk0_2_s42 \
         outputs/20260622_111910_srae_local_local_topk0_2_bg0_0_eps0_125_s42 \
  --victims densenet121 resnet50 mobilenet_v3_large vit_base_patch16_224 deit3_small_patch16_224
```

出:`outputs/transfer_matrix.md` 的 conditional ASR 矩阵。**看 local vs baseline 在 ViT 列的相对差距**——local 明显赢则路线 A 成立且可冲迁移性;赢得不够则考虑路线 B。

## 可选:轻量扫参(B2 若 ASR 卡在 91% 不动时)

```bash
python scripts/make_subset_split.py --per-class 30    # 生成 dataset-trn-n30.txt
python scripts/train_local.py --top-k 0.3 --eps 0.125 \
  --trn-split dataset-trn-n30.txt --lr-drops 20 40 --epochs 60   # ~1h
python scripts/evaluate.py --models-dir outputs/<新run>/models --gated-recovery
```

轻量结果只用于横比排序;胜出配置必须回全量 150ep 重跑才能进论文。

## 推荐执行顺序

B0 → B4(后台挂着训 4 个 victim)→ B1(victim 训练时并行,零训练很快)→ B2 → B3 → B5。
B2/B3 是训练任务,可用 `&` 或分次租用。数据齐了发我一起看。
