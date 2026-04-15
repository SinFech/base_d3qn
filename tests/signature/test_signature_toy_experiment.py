from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "signature_toy_experiment.py"
    spec = importlib.util.spec_from_file_location("signature_toy_experiment", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_toy_dataset_is_balanced_and_has_expected_shapes() -> None:
    module = _load_module()

    windows, labels = module.build_toy_dataset(
        num_per_class=3,
        window_size=12,
        noise_std=0.01,
        seed=11,
    )

    assert len(windows) == 6
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert all(list(window.columns) == ["Close"] for window in windows)
    assert all(window.shape == (12, 1) for window in windows)


def test_signature_encoder_matches_feature_dimension_and_is_finite() -> None:
    module = _load_module()
    windows, _ = module.build_toy_dataset(
        num_per_class=2,
        window_size=16,
        noise_std=0.01,
        seed=21,
    )
    builder, transformer = module.build_signature_encoder(degree=2)

    features = [
        module.compute_signature_features(window, builder, transformer)
        for window in windows
    ]

    expected_dim = transformer.obs_dim(builder.base_dim)
    assert all(feature.shape == (expected_dim,) for feature in features)
    assert np.isfinite(np.stack(features)).all()


def test_run_experiment_returns_required_sections() -> None:
    module = _load_module()

    result = module.run_experiment(
        window_size=12,
        train_per_class=6,
        test_per_class=6,
        noise_std=0.01,
        degree=2,
        data_seed=5,
        train_seed=13,
        epochs=25,
        learning_rate=0.05,
    )

    assert set(result.keys()) == {"config", "class_summaries", "feature_results"}
    assert len(result["class_summaries"]) == 2
    assert [row["feature_set"] for row in result["feature_results"]] == ["summary", "signature"]
    assert all(row["feature_dim"] >= 1 for row in result["feature_results"])
    assert all(0.0 <= row["train_accuracy"] <= 1.0 for row in result["feature_results"])
    assert all(0.0 <= row["test_accuracy"] <= 1.0 for row in result["feature_results"])
