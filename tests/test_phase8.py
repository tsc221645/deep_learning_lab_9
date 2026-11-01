import json

import pandas as pd

from src.experiments.phase8 import _target_fingerprint, expected_runs


def test_phase8_checks_target_values_as_well_as_dates(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    rows = pd.DataFrame({
        "unique_id": ["client"],
        "ds": [pd.Timestamp("2014-10-01")],
        "y_true": [1.0],
    })
    rows.to_parquet(first / "predictions.parquet", index=False)
    rows.assign(y_true=2.0).to_parquet(
        second / "predictions.parquet", index=False
    )
    assert _target_fingerprint(first) != _target_fingerprint(second)


def test_phase8_uses_the_autoets24_artifact_name(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "phase5_selected.json").write_text(json.dumps({
        name: {"config": {"config_id": name}, "validation_mase": 1.0}
        for name in ("lstm_recursive", "lstm_multioutput", "lstm_seq2seq")
    }), encoding="utf-8")
    assert expected_runs(tmp_path)["AutoETS24"] == "auto_ets-h30"
