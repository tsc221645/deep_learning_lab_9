"""Catch incompatible PyTorch vision wheels before a long model run."""

import importlib


def test_neuralforecast_model_imports_with_pinned_torch_stack():
    models = importlib.import_module("neuralforecast.models")
    for name in ("NHITS", "TCN", "PatchTST", "TFT", "DeepAR"):
        assert hasattr(models, name)
