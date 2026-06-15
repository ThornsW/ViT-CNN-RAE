"""Flask app: browse outputs/ runs, compare metrics, view loss/images/logs.

Run dirs are re-discovered on every request (cheap), so a browser refresh picks
up new runs with no restart. Image/log files are streamed via /file with a
path-traversal guard (parse.safe_under); only files under outputs/ are served.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from .. import config
from . import charts, parse

# Columns of the cross-run comparison table; `mode` = which direction is "best".
COMPARE_COLS = [
    {"key": "asr", "label": "ASR %", "mode": "max", "p": 2},
    {"key": "adv_psnr", "label": "Adv PSNR", "mode": "max", "p": 2},
    {"key": "adv_ssim", "label": "Adv SSIM", "mode": "max", "p": 3},
    {"key": "rec_psnr", "label": "Rec PSNR", "mode": "max", "p": 2},
    {"key": "rec_ssim", "label": "Rec SSIM", "mode": "max", "p": 3},
    {"key": "adv_l0_pct", "label": "Adv L0 %", "mode": "min", "p": 1},
    {"key": "adv_linf", "label": "Adv L∞", "mode": "min", "p": 3},
]


def create_app(outputs_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    outputs_dir = Path(outputs_dir) if outputs_dir else config.OUTPUT_DIR
    app.config["OUTPUTS_DIR"] = outputs_dir

    @app.context_processor
    def _inject():
        return {"outputs_dir": str(outputs_dir)}

    @app.route("/")
    def index():
        rows = []
        for r in parse.discover_runs(outputs_dir):
            metrics = parse.parse_metrics(r.path)
            summary = parse.metric_summary(metrics[-1]) if metrics else None
            rows.append({"run": r, "summary": summary, "n_eval": len(metrics)})
        summaries = [row["summary"] for row in rows if row["summary"]]
        best = {}
        for c in COMPARE_COLS:
            vals = [s[c["key"]] for s in summaries if s.get(c["key"]) is not None]
            if vals:
                best[c["key"]] = max(vals) if c["mode"] == "max" else min(vals)
        show_compare = sum(1 for row in rows if row["run"].has_loss) >= 2
        return render_template("index.html", rows=rows, cols=COMPARE_COLS,
                               best=best, show_compare=show_compare, outputs_dir=str(outputs_dir))

    @app.route("/run/<name>")
    def run_detail(name: str):
        r = parse.get_run(name, outputs_dir)
        if r is None:
            abort(404)
        metrics = parse.parse_metrics(r.path)
        summary = parse.metric_summary(metrics[-1]) if metrics else None
        logs = []
        for lp in parse.find_logs(r.path):
            try:
                text = lp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            logs.append({"name": lp.name, "text": text, "size": lp.stat().st_size})
        return render_template(
            "run.html", run=r, summary=summary, n_eval=len(metrics),
            logs=logs, run_config=parse.parse_run_config(r.path),
            description=parse.read_description(r.path),
            run_images=parse.find_run_images(r.path, outputs_dir),
        )

    @app.route("/run/<name>/description", methods=["POST"])
    def save_description(name: str):
        # get_run only matches discovered run dirs, so `name` can't traverse out.
        r = parse.get_run(name, outputs_dir)
        if r is None:
            abort(404)
        text = request.form.get("text")
        if text is None and request.is_json:
            text = (request.get_json(silent=True) or {}).get("text")
        text = text or ""
        if len(text.encode("utf-8")) > 64 * 1024:  # personal note, not a log dump
            abort(413)
        parse.write_description(r.path, text)
        return jsonify(ok=True)

    @app.route("/chart/<name>")
    def chart(name: str):
        r = parse.get_run(name, outputs_dir)
        if r is None:
            abort(404)
        loss = parse.parse_loss(r.path)
        if not loss:
            abort(404)
        mtime = (r.path / "models" / "loss1.txt").stat().st_mtime
        return Response(charts.render_loss_cached(name, mtime, loss), mimetype="image/png")

    @app.route("/compare/loss.png")
    def compare_loss():
        runs, key_parts = [], []
        for r in parse.discover_runs(outputs_dir):
            loss = parse.parse_loss(r.path)
            if loss:
                runs.append((r.tag or r.name, loss))
                key_parts.append((r.name, (r.path / "models" / "loss1.txt").stat().st_mtime))
        if not runs:
            abort(404)
        png = charts.render_compare_cached(("compare", tuple(key_parts)), runs)
        return Response(png, mimetype="image/png")

    @app.route("/gallery")
    def gallery():
        return render_template("gallery.html", images=parse.find_global_images(outputs_dir))

    @app.route("/file")
    def serve_file():
        rel = request.args.get("path", "")
        if not rel:
            abort(404)
        target = outputs_dir / rel
        if not parse.safe_under(outputs_dir, target):
            abort(403)
        if not target.is_file():
            abort(404)
        return send_file(target.resolve())

    return app
