"""Generate a stratified subset of the training split for fast sweeps.

    python scripts/make_subset_split.py --per-class 30
    # -> data/splits/<dataset>/dataset-trn-n30.txt

Why: a full run on Caltech-256 (28k imgs, 150 ep) is 6-10 h. Subsets cut that
to ~1 h so configs can be RANKED cheaply. They are for screening only — every
number that goes in the paper must be re-run on the full split.

Design: stratified (a fixed count per class, not a global random %) so all 257
classes keep enough samples for the class-competitive C&W loss; fixed seed so
the subset is identical across configs and any difference is attributable to
the config, not to the sampling.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-class', type=int, default=30,
                   help='每类抽样张数(不足则全取)')
    p.add_argument('--src', default='dataset-trn.txt', help='源 split 文件名')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    src = config.split_path(args.src)
    lines = [ln.strip() for ln in src.read_text().splitlines() if ln.strip()]

    by_label: dict[str, list[str]] = defaultdict(list)
    for ln in lines:
        by_label[ln.split()[1]].append(ln)

    rng = random.Random(args.seed)
    picked: list[str] = []
    for label in sorted(by_label, key=int):
        pool = by_label[label]
        picked.extend(sorted(rng.sample(pool, min(args.per_class, len(pool)))))

    out = config.split_path(f"dataset-trn-n{args.per_class}.txt")
    out.write_text("\n".join(picked) + "\n")
    print(f"{len(by_label)} classes | {len(lines)} -> {len(picked)} images")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
