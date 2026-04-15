from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "signature_toy_experiment_bitcoin.py"
    spec = importlib.util.spec_from_file_location("signature_toy_experiment_bitcoin", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_volume_profile_preserves_price_and_total_volume() -> None:
    module = _load_module()
    frame = module.load_bitcoin_dataframe()
    windows, _ = module.select_bitcoin_windows(
        frame,
        window_size=12,
        num_windows=1,
        start_date="2017-01-01",
        end_date="2017-12-31",
    )
    source = windows[0]

    early = module.apply_volume_profile(source, front_loaded=True, mix_alpha=0.5)
    late = module.apply_volume_profile(source, front_loaded=False, mix_alpha=0.5)

    assert np.allclose(early["Close"].to_numpy(), late["Close"].to_numpy())
    assert np.allclose(early["High"].to_numpy(), late["High"].to_numpy())
    assert np.allclose(early["Low"].to_numpy(), late["Low"].to_numpy())
    assert np.isclose(early["Volume"].sum(), late["Volume"].sum())
    assert np.isclose(early["Volume"].sum(), source["Volume"].sum())

    half = len(early) // 2
    early_first_half = float(early["Volume"].iloc[:half].sum() / early["Volume"].sum())
    late_first_half = float(late["Volume"].iloc[:half].sum() / late["Volume"].sum())
    assert early_first_half > late_first_half


def test_feature_families_are_finite_and_volume_sensitive() -> None:
    module = _load_module()
    frame = module.load_bitcoin_dataframe()
    windows, _ = module.select_bitcoin_windows(
        frame,
        window_size=16,
        num_windows=1,
        start_date="2018-01-01",
        end_date="2018-12-31",
    )
    source = windows[0]
    early = module.apply_volume_profile(source, front_loaded=True, mix_alpha=0.5)
    late = module.apply_volume_profile(source, front_loaded=False, mix_alpha=0.5)

    early_summary = module.compute_summary_features(early)
    late_summary = module.compute_summary_features(late)
    early_raw = module.compute_raw_window_features(early)
    late_raw = module.compute_raw_window_features(late)
    assert np.isfinite(early_summary).all()
    assert np.isfinite(late_summary).all()
    assert np.isfinite(early_raw).all()
    assert np.isfinite(late_raw).all()
    assert not np.allclose(early_raw, late_raw)

    builder, transformer = module.build_signature_encoder(degree=2)
    early_signature = module.compute_signature_features(early, builder, transformer)
    late_signature = module.compute_signature_features(late, builder, transformer)
    assert np.isfinite(early_signature).all()
    assert np.isfinite(late_signature).all()
    assert not np.allclose(early_signature, late_signature)


def test_run_experiment_returns_required_sections() -> None:
    module = _load_module()

    result = module.run_experiment(
        window_size=16,
        train_per_class=4,
        test_per_class=4,
        train_start_date="2014-01-01",
        train_end_date="2016-12-31",
        test_start_date="2021-01-01",
        test_end_date="2022-12-31",
        degree=2,
        volume_mix_alpha=0.5,
        probe_model="mlp",
        hidden_dim=16,
        train_seed=13,
        epochs=25,
        learning_rate=0.001,
    )

    assert set(result.keys()) == {"config", "data_summary", "class_summaries", "feature_results"}
    assert result["data_summary"]["train"]["selected_windows"] == 4
    assert result["data_summary"]["test"]["selected_windows"] == 4
    assert [row["name"] for row in result["class_summaries"]] == ["early_volume", "late_volume"]
    assert [row["feature_set"] for row in result["feature_results"]] == ["summary", "raw_window", "signature"]
    assert all(row["feature_dim"] >= 1 for row in result["feature_results"])
    assert all(0.0 <= row["train_accuracy"] <= 1.0 for row in result["feature_results"])
    assert all(0.0 <= row["test_accuracy"] <= 1.0 for row in result["feature_results"])
