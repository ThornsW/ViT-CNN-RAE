"""Local web app: browse outputs/ runs, compare metrics, view loss/images/logs.

    /home/thorns1exp/miniforge3/envs/DL/bin/python scripts/report_app.py --port 8000
then open http://127.0.0.1:8000

Needs flask:  pip install -e ".[report]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.report import create_app


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--outputs-dir", type=Path, default=None,
                   help="outputs/ dir to scan (default: config.OUTPUT_DIR)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    app = create_app(outputs_dir=args.outputs_dir)
    print(f"serving outputs report on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
