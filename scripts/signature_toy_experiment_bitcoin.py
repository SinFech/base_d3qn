from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl.envs.make_env import filter_date_range, load_price_data
from rl.features.signature import LogSigTransformer, PathBuilder


DEFAULT_DATA_PATH = Path("data/Bitcoin History 2010-2024.csv")
DEFAULT_DATE_COLUMN = "Date"
CLASS_NAMES = {
    0: "early_volume",
    1: "late_volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Bitcoin-based toy experiment that probes whether logsignature features encode volume timing.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--train-per-class", type=int, default=256)
    parser.add_argument("--test-per-class", type=int, default=256)
    parser.add_argument("--train-start-date", type=str, default="2014-01-01")
    parser.add_argument("--train-end-date", type=str, default="2020-12-31")
    parser.add_argument("--test-start-date", type=str, default="2021-01-01")
    parser.add_argument("--test-end-date", type=str, default="2024-02-09")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--volume-mix-alpha", type=float, default=0.45)
    parser.add_argument("--probe-model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")


def _date_text(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_bitcoin_dataframe(
    data_path: Path = DEFAULT_DATA_PATH,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> pd.DataFrame:
    return load_price_data(data_path, date_column=date_column)


def select_bitcoin_windows(
    df: pd.DataFrame,
    *,
    window_size: int,
    num_windows: int,
    start_date: str | None,
    end_date: str | None,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    _validate_positive("num_windows", num_windows)
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")

    filtered = filter_date_range(
        df,
        date_column=date_column,
        start_date=start_date,
        end_date=end_date,
    )
    if filtered.empty:
        raise ValueError("No rows remain after applying the date filter.")

    candidates: list[pd.DataFrame] = []
    end_dates: list[pd.Timestamp] = []
    for end_index in range(window_size - 1, len(filtered)):
        window = filtered.iloc[end_index - window_size + 1 : end_index + 1].reset_index(drop=True)
        numeric = window[["Close", "High", "Low", "Volume"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            continue
        if float(window["Volume"].sum()) <= 0.0:
            continue
        candidates.append(window)
        end_dates.append(pd.Timestamp(window[date_column].iloc[-1]))

    if len(candidates) < num_windows:
        raise ValueError(
            f"Requested {num_windows} windows but only {len(candidates)} valid windows "
            f"are available in [{start_date}, {end_date}]."
        )

    indices = np.linspace(0, len(candidates) - 1, num=num_windows, dtype=int).tolist()
    selected_windows = [candidates[index] for index in indices]
    selected_end_dates = [end_dates[index] for index in indices]
    summary = {
        "date_range_start": _date_text(pd.Timestamp(filtered[date_column].iloc[0])),
        "date_range_end": _date_text(pd.Timestamp(filtered[date_column].iloc[-1])),
        "available_rows": int(len(filtered)),
        "available_windows": int(len(candidates)),
        "selected_windows": int(len(selected_windows)),
        "selected_first_end_date": _date_text(selected_end_dates[0]),
        "selected_last_end_date": _date_text(selected_end_dates[-1]),
    }
    return selected_windows, summary


def build_volume_profile_weights(
    window_size: int,
    *,
    front_loaded: bool,
    min_weight: float = 1.0,
    max_weight: float = 2.0,
) -> np.ndarray:
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")
    if min_weight <= 0.0 or max_weight <= 0.0:
        raise ValueError("Volume profile weights must be positive.")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight.")

    weights = np.linspace(max_weight, min_weight, window_size, dtype=np.float64)
    if not front_loaded:
        weights = weights[::-1]
    return weights / weights.sum()


def apply_volume_profile(
    window: pd.DataFrame,
    *,
    front_loaded: bool,
    mix_alpha: float = 0.45,
) -> pd.DataFrame:
    total_volume = float(window["Volume"].sum())
    if not np.isfinite(total_volume) or total_volume <= 0.0:
        raise ValueError("Window volume must be finite and positive.")
    if not 0.0 <= mix_alpha <= 1.0:
        raise ValueError("mix_alpha must be in [0.0, 1.0].")

    adjusted = window.copy()
    original = window["Volume"].to_numpy(dtype=np.float64)
    original = original / max(float(original.sum()), 1e-12)
    target = build_volume_profile_weights(
        len(window),
        front_loaded=front_loaded,
    )
    mixed = (1.0 - mix_alpha) * original + mix_alpha * target
    mixed = mixed / max(float(mixed.sum()), 1e-12)
    adjusted["Volume"] = total_volume * mixed
    return adjusted


def build_paired_volume_dataset(
    source_windows: list[pd.DataFrame],
    *,
    mix_alpha: float,
) -> tuple[list[pd.DataFrame], np.ndarray]:
    windows: list[pd.DataFrame] = []
    labels: list[int] = []
    for window in source_windows:
        windows.append(apply_volume_profile(window, front_loaded=True, mix_alpha=mix_alpha))
        labels.append(0)
        windows.append(apply_volume_profile(window, front_loaded=False, mix_alpha=mix_alpha))
        labels.append(1)
    return windows, np.asarray(labels, dtype=np.int64)


def compute_summary_features(window: pd.DataFrame) -> np.ndarray:
    close = window["Close"].to_numpy(dtype=np.float64)
    log_close = np.log(np.clip(close, 1e-8, None))
    log_return = np.zeros_like(log_close)
    if log_close.size > 1:
        log_return[1:] = log_close[1:] - log_close[:-1]
    volume = window["Volume"].to_numpy(dtype=np.float64)
    return np.asarray(
        [
            log_close[-1] - log_close[0],
            log_return.mean(),
            log_return.std(),
            close.max(),
            close.min(),
            close.max() - close.min(),
            volume.sum(),
            volume.mean(),
            volume.std(),
        ],
        dtype=np.float32,
    )


def compute_raw_window_features(window: pd.DataFrame) -> np.ndarray:
    return (
        window[["Close", "High", "Low", "Volume"]]
        .to_numpy(dtype=np.float32)
        .reshape(-1)
        .astype(np.float32, copy=False)
    )


def build_signature_encoder(degree: int) -> tuple[PathBuilder, LogSigTransformer]:
    builder = PathBuilder(
        embedding={
            "log_price": {},
            "log_return": {},
            "normalized_cumulative_volume": {},
            "high_low_range": {},
        }
    )
    transformer = LogSigTransformer(
        degree=degree,
        method=1,
        time_aug=True,
        lead_lag=False,
        end_time=1.0,
        n_jobs=1,
        device="cpu",
        dtype="float32",
        use_disk_prepare_cache=False,
        base_dim=builder.base_dim,
    )
    return builder, transformer


def compute_signature_features(
    window: pd.DataFrame,
    builder: PathBuilder,
    transformer: LogSigTransformer,
) -> np.ndarray:
    return transformer(builder(window)).detach().cpu().numpy().astype(np.float32, copy=False)


def _stack_features(
    windows: list[pd.DataFrame],
    feature_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    return np.stack([feature_fn(window) for window in windows]).astype(np.float32, copy=False)


def _normalize_train_test(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_norm = ((train_features - mean) / std).astype(np.float32, copy=False)
    test_norm = ((test_features - mean) / std).astype(np.float32, copy=False)
    return train_norm, test_norm


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def train_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    model_type: str,
    hidden_dim: int,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float | int]:
    _validate_positive("epochs", epochs)
    _validate_positive("hidden_dim", hidden_dim)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")

    train_x, test_x = _normalize_train_test(train_features, test_features)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    if model_type == "linear":
        model: nn.Module = nn.Linear(train_x.shape[1], 1)
    elif model_type == "mlp":
        model = MLPProbe(train_x.shape[1], hidden_dim)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_labels.astype(np.float32)).unsqueeze(1)
    x_test = torch.from_numpy(test_x)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = (torch.sigmoid(model(x_train)).squeeze(1).numpy() >= 0.5).astype(np.int64)
        test_pred = (torch.sigmoid(model(x_test)).squeeze(1).numpy() >= 0.5).astype(np.int64)

    return {
        "feature_dim": int(train_x.shape[1]),
        "train_accuracy": float((train_pred == train_labels).mean()),
        "test_accuracy": float((test_pred == test_labels).mean()),
    }


def _share(volume: np.ndarray, start: int, end: int) -> float:
    total = float(volume.sum())
    if total <= 0.0:
        return 0.0
    return float(volume[start:end].sum() / total)


def summarize_class_windows(windows: list[pd.DataFrame], label: int) -> dict[str, object]:
    volumes = np.stack([window["Volume"].to_numpy(dtype=np.float64) for window in windows])
    half = volumes.shape[1] // 2
    return {
        "label": int(label),
        "name": CLASS_NAMES[int(label)],
        "mean_total_volume": float(volumes.sum(axis=1).mean()),
        "mean_volume_std": float(volumes.std(axis=1).mean()),
        "mean_first_half_share": float(np.mean([_share(volume, 0, half) for volume in volumes])),
        "mean_second_half_share": float(np.mean([_share(volume, half, len(volume)) for volume in volumes])),
    }


def run_experiment(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    window_size: int = 24,
    train_per_class: int = 256,
    test_per_class: int = 256,
    train_start_date: str = "2014-01-01",
    train_end_date: str = "2020-12-31",
    test_start_date: str = "2021-01-01",
    test_end_date: str = "2024-02-09",
    degree: int = 2,
    volume_mix_alpha: float = 0.45,
    probe_model: str = "mlp",
    hidden_dim: int = 32,
    train_seed: int = 17,
    epochs: int = 400,
    learning_rate: float = 0.001,
) -> dict[str, object]:
    frame = load_bitcoin_dataframe(data_path)
    train_source_windows, train_source_summary = select_bitcoin_windows(
        frame,
        window_size=window_size,
        num_windows=train_per_class,
        start_date=train_start_date,
        end_date=train_end_date,
    )
    test_source_windows, test_source_summary = select_bitcoin_windows(
        frame,
        window_size=window_size,
        num_windows=test_per_class,
        start_date=test_start_date,
        end_date=test_end_date,
    )

    train_windows, train_labels = build_paired_volume_dataset(
        train_source_windows,
        mix_alpha=volume_mix_alpha,
    )
    test_windows, test_labels = build_paired_volume_dataset(
        test_source_windows,
        mix_alpha=volume_mix_alpha,
    )

    builder, transformer = build_signature_encoder(degree=degree)
    summary_train = _stack_features(train_windows, compute_summary_features)
    summary_test = _stack_features(test_windows, compute_summary_features)
    raw_train = _stack_features(train_windows, compute_raw_window_features)
    raw_test = _stack_features(test_windows, compute_raw_window_features)
    signature_train = _stack_features(
        train_windows,
        lambda window: compute_signature_features(window, builder, transformer),
    )
    signature_test = _stack_features(
        test_windows,
        lambda window: compute_signature_features(window, builder, transformer),
    )

    class_windows = {
        0: train_windows[0::2] + test_windows[0::2],
        1: train_windows[1::2] + test_windows[1::2],
    }
    class_summaries = [
        summarize_class_windows(class_windows[label], label)
        for label in sorted(class_windows)
    ]

    feature_results = []
    for name, train_features, test_features in (
        ("summary", summary_train, summary_test),
        ("raw_window", raw_train, raw_test),
        ("signature", signature_train, signature_test),
    ):
        metrics = train_probe(
            train_features,
            train_labels,
            test_features,
            test_labels,
            model_type=probe_model,
            hidden_dim=hidden_dim,
            seed=train_seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        feature_results.append({"feature_set": name, **metrics})

    return {
        "config": {
            "data_path": str(data_path),
            "window_size": int(window_size),
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "degree": int(degree),
            "volume_mix_alpha": float(volume_mix_alpha),
            "probe_model": probe_model,
            "hidden_dim": int(hidden_dim),
            "train_seed": int(train_seed),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
        },
        "data_summary": {
            "train": train_source_summary,
            "test": test_source_summary,
        },
        "class_summaries": class_summaries,
        "feature_results": feature_results,
    }


def format_experiment_report(result: dict[str, object]) -> str:
    config = result["config"]
    data_summary = result["data_summary"]
    class_summaries = result["class_summaries"]
    feature_results = result["feature_results"]

    lines = ["config:"]
    for key, value in config.items():
        lines.append(f"  {key}={value}")

    lines.append("")
    lines.append("data_summary:")
    for split_name in ("train", "test"):
        summary = data_summary[split_name]
        lines.append(
            "  "
            + f"{split_name}: "
            + f"date_range=[{summary['date_range_start']}, {summary['date_range_end']}], "
            + f"available_rows={summary['available_rows']}, "
            + f"available_windows={summary['available_windows']}, "
            + f"selected_windows={summary['selected_windows']}, "
            + f"selected_end_dates=[{summary['selected_first_end_date']}, {summary['selected_last_end_date']}]"
        )

    lines.append("")
    lines.append("class_summaries:")
    for summary in class_summaries:
        lines.append(
            "  "
            + f"{summary['name']}: "
            + f"mean_total_volume={summary['mean_total_volume']:.4f}, "
            + f"mean_volume_std={summary['mean_volume_std']:.4f}, "
            + f"mean_first_half_share={summary['mean_first_half_share']:.4f}, "
            + f"mean_second_half_share={summary['mean_second_half_share']:.4f}"
        )

    lines.append("")
    lines.append("probe_results:")
    header = f"{'feature_set':<12} {'dim':>4} {'train_acc':>10} {'test_acc':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in feature_results:
        lines.append(
            f"{row['feature_set']:<12} "
            f"{row['feature_dim']:>4d} "
            f"{row['train_accuracy']:>10.4f} "
            f"{row['test_accuracy']:>10.4f}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_experiment(
        data_path=args.data_path,
        window_size=args.window_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        degree=args.degree,
        volume_mix_alpha=args.volume_mix_alpha,
        probe_model=args.probe_model,
        hidden_dim=args.hidden_dim,
        train_seed=args.train_seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(format_experiment_report(result))


if __name__ == "__main__":
    main()
