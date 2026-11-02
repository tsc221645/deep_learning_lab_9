import base64
import json
import re
import zlib

import pandas as pd

from src.experiments.report_status import (
    EXPECTED_MODELS,
    _write_pdf,
    collect_smoke_runs,
)


def _write_smoke(path, name, config_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": name,
                "config_id": config_id,
                "metrics": {"n_observations": 192, "n_series": 8},
                "hardware": {"device": "cuda"},
            }
        ),
        encoding="utf-8",
    )


def test_collect_smokes_reports_recorded_devices(tmp_path):
    outputs = tmp_path / "outputs"
    names = sorted(EXPECTED_MODELS)
    for index, name in enumerate(names):
        root = (
            outputs / "predictions"
            if index < 8
            else outputs / "smoke_b" / "predictions"
        )
        _write_smoke(root / "model" / f"{name.lower()}-smoke" / "experiment.json",
                     name, f"{name.lower()}-smoke")

    runs = collect_smoke_runs(tmp_path)

    assert set(runs["model"]) == EXPECTED_MODELS
    assert runs["predictions"].eq(192).all()
    assert runs["origins"].eq(1).all()
    assert runs["device"].eq("cuda").all()


def test_collect_smokes_preserves_cpu_device_metadata(tmp_path):
    outputs = tmp_path / "outputs" / "predictions"
    for name in EXPECTED_MODELS:
        _write_smoke(
            outputs / "model" / f"{name.lower()}-smoke" / "experiment.json",
            name,
            f"{name.lower()}-smoke",
        )
    path = (
        outputs / "model" / "naive-smoke" / "experiment.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["hardware"]["device"] = "cpu"
    path.write_text(json.dumps(record), encoding="utf-8")

    runs = collect_smoke_runs(tmp_path)

    assert runs.loc[runs["model"].eq("Naive"), "device"].item() == "cpu"
    assert runs["device"].eq("cuda").sum() == len(EXPECTED_MODELS) - 1


def test_status_pdf_has_clear_nonfinal_label_and_five_page_limit(tmp_path):
    runs = pd.DataFrame(
        [
            {
                "model": name,
                "clients": 8,
                "origins": 1,
                "predictions": 192,
                "device": "cuda",
            }
            for name in sorted(EXPECTED_MODELS)
        ]
    )
    pdf_path = tmp_path / "informe_estado.pdf"
    _write_pdf(
        pdf_path,
        runs,
        "https://github.com/example/project",
        "2026-11-01",
        43,
        ["outputs/metrics/phase4_summary.csv"],
        "test",
    )
    content = pdf_path.read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)endstream", content, re.DOTALL)
    uncompressed = []
    for stream in streams:
        try:
            encoded = stream.strip().removesuffix(b"~>")
            uncompressed.append(zlib.decompress(base64.a85decode(encoded)))
        except (ValueError, zlib.error):
            uncompressed.append(stream)

    assert content.startswith(b"%PDF-")
    assert b"NO INFORME FINAL" in b"\n".join(uncompressed)
    assert len(re.findall(rb"/Type\s*/Page\b", content)) <= 5
