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

from rl.features.signature import LogSigTransformer, PathBuilder


CLASS_NAMES = {
    0: "up_first",
    1: "down_first",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a toy experiment that probes whether logsignature features encode path geometry.",
    )
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--train-per-class", type=int, default=256)
    parser.add_argument("--test-per-class", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")


def build_toy_dataset(
    num_per_class: int,
    window_size: int,
    noise_std: float,
    seed: int,
    *,
    base_price: float = 100.0,
    amplitude: float = 0.15,
) -> tuple[list[pd.DataFrame], np.ndarray]:
    _validate_positive("num_per_class", num_per_class)
    if window_size < 3:
        raise ValueError("window_size must be >= 3.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")

    rng = np.random.default_rng(seed)
    phase = np.sin(np.linspace(0.0, 2.0 * np.pi, window_size))
    base_log_price = np.log(base_price)

    windows: list[pd.DataFrame] = []
    labels: list[int] = []
    for label, sign in ((0, 1.0), (1, -1.0)):
        for _ in range(num_per_class):
            noise = rng.normal(0.0, noise_std, size=window_size)
            log_close = base_log_price + sign * amplitude * phase + noise
            close = np.exp(log_close)
            windows.append(pd.DataFrame({"Close": close}))
            labels.append(label)
    return windows, np.asarray(labels, dtype=np.int64)


def compute_summary_features(window: pd.DataFrame) -> np.ndarray:
    close = window["Close"].to_numpy(dtype=np.float64)
    log_close = np.log(np.clip(close, 1e-8, None))
    log_return = np.zeros_like(log_close)
    if log_close.size > 1:
        log_return[1:] = log_close[1:] - log_close[:-1]
    return np.asarray(
        [
            log_close[-1] - log_close[0],
            log_return.mean(),
            log_return.std(),
            close.max(),
            close.min(),
            close.max() - close.min(),
        ],
        dtype=np.float32,
    )


def build_signature_encoder(degree: int) -> tuple[PathBuilder, LogSigTransformer]:
    builder = PathBuilder(
        embedding={
            "log_price": {},
            "log_return": {},
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


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float | int]:
    _validate_positive("epochs", epochs)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")

    train_x, test_x = _normalize_train_test(train_features, test_features)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    model = nn.Linear(train_x.shape[1], 1)
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


def summarize_class_windows(windows: list[pd.DataFrame], label: int) -> dict[str, object]:
    stacked = np.stack([window["Close"].to_numpy(dtype=np.float64) for window in windows])
    mean_path = stacked.mean(axis=0)
    max_values = stacked.max(axis=1)
    min_values = stacked.min(axis=1)
    log_end_return = np.log(np.clip(stacked[:, -1], 1e-8, None)) - np.log(np.clip(stacked[:, 0], 1e-8, None))
    return {
        "label": int(label),
        "name": CLASS_NAMES[int(label)],
        "mean_log_end_return": float(log_end_return.mean()),
        "mean_max_close": float(max_values.mean()),
        "mean_min_close": float(min_values.mean()),
        "mean_range": float((max_values - min_values).mean()),
        "mean_path_head": [float(value) for value in mean_path[:5]],
        "mean_path_tail": [float(value) for value in mean_path[-5:]],
    }


def run_experiment(
    *,
    window_size: int = 24,
    train_per_class: int = 256,
    test_per_class: int = 256,
    noise_std: float = 0.01,
    degree: int = 2,
    data_seed: int = 7,
    train_seed: int = 17,
    epochs: int = 400,
    learning_rate: float = 0.05,
) -> dict[str, object]:
    train_windows, train_labels = build_toy_dataset(
        num_per_class=train_per_class,
        window_size=window_size,
        noise_std=noise_std,
        seed=data_seed,
    )
    test_windows, test_labels = build_toy_dataset(
        num_per_class=test_per_class,
        window_size=window_size,
        noise_std=noise_std,
        seed=data_seed + 1,
    )

    builder, transformer = build_signature_encoder(degree=degree)
    summary_train = _stack_features(train_windows, compute_summary_features)
    summary_test = _stack_features(test_windows, compute_summary_features)
    signature_train = _stack_features(
        train_windows,
        lambda window: compute_signature_features(window, builder, transformer),
    )
    signature_test = _stack_features(
        test_windows,
        lambda window: compute_signature_features(window, builder, transformer),
    )

    class_windows = {
        0: train_windows[:train_per_class] + test_windows[:test_per_class],
        1: train_windows[train_per_class:] + test_windows[test_per_class:],
    }
    class_summaries = [
        summarize_class_windows(class_windows[label], label)
        for label in sorted(class_windows)
    ]

    feature_results = []
    for name, train_features, test_features in (
        ("summary", summary_train, summary_test),
        ("signature", signature_train, signature_test),
    ):
        metrics = train_linear_probe(
            train_features,
            train_labels,
            test_features,
            test_labels,
            seed=train_seed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        feature_results.append({"feature_set": name, **metrics})

    return {
        "config": {
            "window_size": int(window_size),
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "noise_std": float(noise_std),
            "degree": int(degree),
            "data_seed": int(data_seed),
            "train_seed": int(train_seed),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
        },
        "class_summaries": class_summaries,
        "feature_results": feature_results,
    }


def _format_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def format_experiment_report(result: dict[str, object]) -> str:
    config = result["config"]
    class_summaries = result["class_summaries"]
    feature_results = result["feature_results"]

    lines = ["config:"]
    for key, value in config.items():
        lines.append(f"  {key}={value}")

    lines.append("")
    lines.append("class_summaries:")
    for summary in class_summaries:
        lines.append(
            "  "
            + f"{summary['name']}: "
            + f"mean_log_end_return={summary['mean_log_end_return']:.6f}, "
            + f"mean_max_close={summary['mean_max_close']:.4f}, "
            + f"mean_min_close={summary['mean_min_close']:.4f}, "
            + f"mean_range={summary['mean_range']:.4f}"
        )
        lines.append(f"    mean_path_head={_format_float_list(summary['mean_path_head'])}")
        lines.append(f"    mean_path_tail={_format_float_list(summary['mean_path_tail'])}")

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
        window_size=args.window_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        noise_std=args.noise_std,
        degree=args.degree,
        data_seed=args.data_seed,
        train_seed=args.train_seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(format_experiment_report(result))


if __name__ == "__main__":
    main()
