import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_pipeline_runs(tmp_path):
    from testing.pipeline import full_pipeline
    out = tmp_path / "out"
    full_pipeline("data/gri30.yaml", str(out), steps=100, tf=1.0)
    assert (out / "metrics.csv").exists()
