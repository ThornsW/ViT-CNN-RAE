"""CPU-only smoke tests for the report layer (no torch).

Parse-layer tests rely on the run dirs already present under outputs/; the
description tests also drive the Flask routes via a test client (flask is a
report dependency, so no torch is pulled in).
"""
from pathlib import Path

from vit_cnn_rae import config
from vit_cnn_rae.report import parse

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
BASELINE = "20260602_024646_srae_baseline_s42"


def test_discover_runs_finds_known_run():
    names = {r.name for r in parse.discover_runs(OUTPUTS)}
    assert BASELINE in names
    # shared buckets under outputs/ are not runs
    assert "attention_viz" not in names
    assert "models" not in names


def test_parse_run_name():
    meta = parse.parse_run_name(BASELINE)
    assert meta["model"] == "srae"
    assert meta["tag"] == "baseline"
    assert meta["seed"] == 42


def test_parse_metrics_and_summary():
    metrics = parse.parse_metrics(OUTPUTS / BASELINE)
    assert metrics, "baseline test.txt should parse to >=1 eval dict"
    s = parse.metric_summary(metrics[-1])
    assert 98.0 < s["asr"] < 99.5          # ASR ~98.75
    assert 25.0 < s["adv_psnr"] < 40.0     # ~30.79 dB
    assert 0.0 < s["adv_l0_pct"] <= 100.0


def test_parse_loss_six_series():
    loss = parse.parse_loss(OUTPUTS / BASELINE)
    assert len(loss) == 6
    lengths = {len(v) for v in loss.values()}
    assert len(lengths) == 1 and lengths.pop() == 150


def test_safe_under_blocks_traversal():
    assert parse.safe_under(OUTPUTS, OUTPUTS / "a" / "b.png")
    assert not parse.safe_under(OUTPUTS, OUTPUTS / ".." / ".." / "etc" / "passwd")


def test_run_config_roundtrip(tmp_path):
    params = {"seed": 42, "top_k": 0.2, "resume": None, "out": Path("/x/y")}
    f = config.save_run_config(tmp_path, params, model="srae_local")
    assert f.is_file()
    rc = parse.parse_run_config(tmp_path)
    assert rc is not None
    assert rc["model"] == "srae_local"
    assert rc["params"]["seed"] == 42
    assert rc["params"]["resume"] is None
    assert rc["params"]["out"] == "/x/y"     # Path coerced to str
    assert "started_at" in rc


def test_parse_run_config_absent():
    assert parse.parse_run_config(OUTPUTS / BASELINE) is None


def test_compare_figure_is_png():
    from vit_cnn_rae.report import charts
    runs = [("a", {"loss_D": [1.0, 0.5, 0.2]}), ("b", {"loss_D": [1.1, 0.6, 0.3]})]
    assert charts.render_compare_figure(runs)[:4] == b"\x89PNG"


def test_global_images_are_outputs_relative():
    imgs = parse.find_global_images(OUTPUTS)
    # attention_viz holds PNGs; paths come back relative to outputs/
    assert any(p.startswith("attention_viz/") for p in imgs["attention"])


def test_description_absent_returns_empty():
    assert parse.read_description(OUTPUTS / BASELINE) == ""


def test_description_roundtrip(tmp_path):
    parse.write_description(tmp_path, "测 top-k=0.1 在 ViT 注意力下的 PSNR")
    assert parse.read_description(tmp_path) == "测 top-k=0.1 在 ViT 注意力下的 PSNR"
    # blank text clears the sidecar entirely
    parse.write_description(tmp_path, "   ")
    assert parse.read_description(tmp_path) == ""
    assert not (tmp_path / "description.txt").exists()


def _demo_run(outputs_dir: Path) -> Path:
    run = outputs_dir / "20260101_000000_srae_demo_s1"
    (run / "models").mkdir(parents=True)
    return run


def test_description_route_persists_and_renders(tmp_path):
    from vit_cnn_rae.report.app import create_app
    run = _demo_run(tmp_path)
    client = create_app(tmp_path).test_client()

    res = client.post(f"/run/{run.name}/description", data={"text": "记录：测试 X"})
    assert res.status_code == 200 and res.get_json()["ok"] is True
    assert (run / "description.txt").read_text(encoding="utf-8") == "记录：测试 X"

    # the saved note shows up when the run page is rendered
    page = client.get(f"/run/{run.name}")
    assert "记录：测试 X" in page.get_data(as_text=True)


def test_description_route_unknown_run_404(tmp_path):
    from vit_cnn_rae.report.app import create_app
    client = create_app(tmp_path).test_client()
    res = client.post("/run/does_not_exist/description", data={"text": "x"})
    assert res.status_code == 404
