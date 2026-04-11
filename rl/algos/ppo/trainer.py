from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from rl.algos.ppo.networks import GaussianActorCriticMLP
from rl.envs.make_env import filter_date_range, load_price_data, make_env, sample_train_test_split
from rl.utils.checkpoint import load_checkpoint, save_checkpoint
from rl.utils.logging import LogPaths, MetricsLogger, setup_run_logger
from rl.utils.path import RunPaths
from rl.utils.seed import seed_everything


@dataclass
class DataConfig:
    path: str = "data/Bitcoin History 2010-2024.csv"
    start_date: Optional[str] = "2016-07-09"
    end_date: Optional[str] = "2018-01-09"
    price_column: str = "Price"
    close_column: str = "Close"
    date_column: str = "Date"


@dataclass
class LogSigConfig:
    degree: int = 2
    method: int = 1
    time_aug: bool = True
    lead_lag: bool = False
    end_time: float = 1.0


@dataclass
class SignatureTorchConfig:
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class SignaturePerfConfig:
    n_jobs: int = -1
    use_disk_prepare_cache: bool = True
    prepare_cache_dir: str = "data/pysiglib_prepare_cache"


@dataclass
class SignatureObsConfig:
    backend: str = "pysiglib"
    embedding: dict = field(default_factory=lambda: {"log_price": {}, "log_return": {}})
    rolling_mean_window: int = 5
    standardize_path_channels: bool = False
    basepoint: bool = False
    subwindow_sizes: list[int] = field(default_factory=list)
    account_features: list[str] = field(default_factory=list)
    logsig: LogSigConfig = field(default_factory=LogSigConfig)
    torch: SignatureTorchConfig = field(default_factory=SignatureTorchConfig)
    perf: SignaturePerfConfig = field(default_factory=SignaturePerfConfig)


@dataclass
class ObsConfig:
    type: str = "raw"
    signature: SignatureObsConfig = field(default_factory=SignatureObsConfig)


@dataclass
class EnvConfig:
    reward: str = "sr_enhanced"
    window_size: int = 24
    trading_period: int = 500
    train_split: float = 0.8
    action_mode: str = "continuous"
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 2.0
    allow_short: bool = False
    max_leverage: float = 1.0
    action_low: float = 0.0
    action_high: float = 1.0
    min_equity_ratio: float = 0.2
    stop_on_bankruptcy: bool = True
    # Keep legacy discrete fields for backward compatibility.
    max_positions: Optional[int] = None
    sell_mode: str = "all"
    obs: ObsConfig = field(default_factory=ObsConfig)


@dataclass
class PPOConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    normalize_advantage: bool = True
    action_dim: int = 1
    init_log_std: float = -1.0
    deterministic_eval: bool = True
    # Legacy field for compatibility with older configs.
    action_number: int = 3


@dataclass
class TrainConfig:
    num_episodes: int = 100
    max_steps_per_episode: Optional[int] = None
    max_total_steps: Optional[int] = None
    log_interval: int = 10
    checkpoint_interval: int = 5
    resample_train_window_each_episode: bool = False
    # Keep compatibility with existing D3QN config files.
    eval_epsilon: float = 0.0
    eval_interval: int = 0
    eval_episodes: int = 0
    eval_seed: Optional[int] = None
    plateau_threshold: float = 0.0
    plateau_patience: int = 0


@dataclass
class EvalConfig:
    num_episodes: int = 50
    seed: int = 20240101
    fixed_windows: bool = True
    fixed_windows_seed: Optional[int] = None
    epsilon: float = 0.0
    save_per_episode: bool = True


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    data: DataConfig
    env: EnvConfig
    ppo: PPOConfig
    train: TrainConfig
    eval: EvalConfig
    run: RunConfig
    algo: str = "ppo_continuous"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _compute_start_range(
    data_len: int,
    window_size: int,
    trading_period: Optional[int],
) -> Tuple[int, int]:
    min_start = window_size - 1
    max_start = data_len - 1 if trading_period is None else data_len - trading_period
    return min_start, max_start


def _legacy_ppo_payload(raw: Dict) -> Dict:
    payload: Dict = {}
    raw_agent = raw.get("agent", {}) or {}
    raw_model = raw.get("model", {}) or {}
    if "action_number" in raw_agent:
        payload["action_number"] = raw_agent["action_number"]
    if "learning_rate" in raw_agent:
        payload["learning_rate"] = raw_agent["learning_rate"]
    if "gamma" in raw_agent:
        payload["gamma"] = raw_agent["gamma"]
    if raw_model.get("hidden_sizes") is not None:
        payload["hidden_sizes"] = raw_model["hidden_sizes"]
    return payload


def config_from_dict(data: Dict) -> Config:
    raw_env = data.get("env", {})
    raw_obs = raw_env.get("obs", {})
    raw_signature = raw_obs.get("signature", {}) or {}
    raw_logsig = raw_signature.get("logsig", {}) or {}
    raw_signature_torch = raw_signature.get("torch", {}) or {}
    raw_signature_perf = raw_signature.get("perf", {}) or {}

    signature_cfg = SignatureObsConfig(
        backend=raw_signature.get("backend", "pysiglib"),
        embedding=raw_signature.get("embedding", {"log_price": {}, "log_return": {}}),
        rolling_mean_window=raw_signature.get("rolling_mean_window", 5),
        standardize_path_channels=raw_signature.get("standardize_path_channels", False),
        basepoint=raw_signature.get("basepoint", False),
        subwindow_sizes=raw_signature.get("subwindow_sizes", []),
        account_features=raw_signature.get("account_features", []),
        logsig=LogSigConfig(**raw_logsig),
        torch=SignatureTorchConfig(**raw_signature_torch),
        perf=SignaturePerfConfig(**raw_signature_perf),
    )
    obs_cfg = ObsConfig(type=raw_obs.get("type", "raw"), signature=signature_cfg)
    env_payload = {key: value for key, value in raw_env.items() if key != "obs"}

    merged_ppo = _legacy_ppo_payload(data)
    merged_ppo.update(data.get("ppo", {}) or {})
    config = Config(
        data=DataConfig(**data.get("data", {})),
        env=EnvConfig(**env_payload, obs=obs_cfg),
        ppo=PPOConfig(**merged_ppo),
        train=TrainConfig(**data.get("train", {})),
        eval=EvalConfig(**data.get("eval", {})),
        run=RunConfig(**data.get("run", {})),
        algo=str(data.get("algo", "ppo_continuous")),
    )
    _validate_config(config)
    return config


def config_to_dict(config: Config) -> Dict:
    return asdict(config)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    return config_from_dict(raw)


def save_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))


def _validate_config(config: Config) -> None:
    if config.env.action_mode != "continuous":
        raise ValueError("PPO trainer requires env.action_mode='continuous'.")
    if config.ppo.action_dim != 1:
        raise ValueError("Current continuous trading environment expects ppo.action_dim == 1.")
    if config.env.action_high <= config.env.action_low:
        raise ValueError("env.action_high must be greater than env.action_low.")
    if (not config.env.allow_short) and config.env.action_low < 0:
        raise ValueError("env.action_low must be >= 0 when allow_short is false.")
    if config.env.initial_capital <= 0:
        raise ValueError("env.initial_capital must be > 0.")


def _prepare_data(config: Config):
    df = load_price_data(
        Path(config.data.path),
        price_column=config.data.price_column,
        close_column=config.data.close_column,
        date_column=config.data.date_column,
    )
    df = filter_date_range(
        df,
        date_column=config.data.date_column,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )
    return df


def _unwrap_base_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "equity_start") or hasattr(current, "agent_positions"):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return None


def _to_obs_tensor(obs, device: str) -> Optional[torch.Tensor]:
    if obs is None:
        return None
    if isinstance(obs, torch.Tensor):
        out = obs.to(device=device, dtype=torch.float32)
    else:
        out = torch.as_tensor(obs, device=device, dtype=torch.float32)
    return out.reshape(1, -1)


def _action_bounds(config: Config, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.full((config.ppo.action_dim,), float(config.env.action_low), device=device, dtype=torch.float32)
    high = torch.full((config.ppo.action_dim,), float(config.env.action_high), device=device, dtype=torch.float32)
    return low, high


def _scale_action(
    squashed_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    return action_low + 0.5 * (squashed_action + 1.0) * (action_high - action_low)


def _squashed_log_prob(
    dist,
    raw_action: torch.Tensor,
    squashed_action: torch.Tensor,
) -> torch.Tensor:
    log_prob = dist.log_prob(raw_action)
    correction = torch.log(1.0 - squashed_action.pow(2) + 1e-6)
    return (log_prob - correction).sum(dim=-1)


def _env_action_from_tensor(action: torch.Tensor) -> float:
    values = action.detach().cpu().numpy().reshape(-1)
    if values.size == 0:
        return 0.0
    return float(values[0])


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(last_value, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def _build_model(config: Config, obs_dim: int, device: str) -> GaussianActorCriticMLP:
    return GaussianActorCriticMLP(
        obs_dim=obs_dim,
        action_dim=config.ppo.action_dim,
        hidden_sizes=config.ppo.hidden_sizes,
        init_log_std=config.ppo.init_log_std,
    ).to(device)


def _compute_episode_return_rate(base_env) -> float:
    if base_env is None:
        return 0.0
    if hasattr(base_env, "equity_start") and hasattr(base_env, "equity_end"):
        equity_start = float(base_env.equity_start)
        equity_end = float(base_env.equity_end)
        return (equity_end / (equity_start + 1e-8)) - 1.0
    if (
        hasattr(base_env, "init_price")
        and hasattr(base_env, "realized_pnl")
        and hasattr(base_env, "agent_open_position_value")
    ):
        equity_start = float(base_env.init_price)
        equity_end = equity_start + float(base_env.realized_pnl) + float(base_env.agent_open_position_value)
        return (equity_end / (equity_start + 1e-8)) - 1.0
    return 0.0


def _build_env(
    config: Config,
    df,
    device: str,
    trading_period: Optional[int] = None,
):
    return make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        trading_period=trading_period,
        max_positions=config.env.max_positions,
        sell_mode=config.env.sell_mode,
        action_mode=config.env.action_mode,
        initial_capital=config.env.initial_capital,
        transaction_cost_bps=config.env.transaction_cost_bps,
        slippage_bps=config.env.slippage_bps,
        allow_short=config.env.allow_short,
        max_leverage=config.env.max_leverage,
        action_low=config.env.action_low,
        action_high=config.env.action_high,
        min_equity_ratio=config.env.min_equity_ratio,
        stop_on_bankruptcy=config.env.stop_on_bankruptcy,
        obs_config=config.env.obs,
    )


def _evaluate_with_model(
    config: Config,
    model: GaussianActorCriticMLP,
    episodes: int,
    epsilon: float,
    device: str,
    eval_seed: Optional[int] = None,
    eval_indices: Optional[List[int]] = None,
    eval_df=None,
) -> Tuple[float, Dict[str, float], list]:
    seed = eval_seed if eval_seed is not None else config.run.seed
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    df = _prepare_data(config) if eval_df is None else eval_df
    min_start, max_start = _compute_start_range(
        len(df),
        config.env.window_size,
        config.env.trading_period,
    )
    if max_start < min_start:
        raise ValueError(
            "Invalid start_index range "
            f"[{min_start}, {max_start}] for data length {len(df)} "
            f"and trading_period {config.env.trading_period}."
        )

    if eval_indices is not None:
        if len(eval_indices) < episodes:
            raise ValueError("eval_indices length must be >= episodes.")
        start_indices = [int(eval_indices[i]) for i in range(episodes)]
    else:
        start_indices = rng.integers(min_start, max_start + 1, size=episodes).tolist()

    env = _build_env(config, df, device, trading_period=config.env.trading_period)
    action_low, action_high = _action_bounds(config, device)

    returns: List[float] = []
    return_rates: List[float] = []
    cumulative_returns: List[list] = []
    initial_state_none_episodes = 0
    zero_step_episodes = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start_index in start_indices:
                env.reset(start_index=int(start_index))
                state = env.get_state()
                if state is None:
                    initial_state_none_episodes += 1
                    returns.append(0.0)
                    return_rates.append(0.0)
                    cumulative_returns.append(getattr(env, "cumulative_return", []))
                    continue

                episode_return = 0.0
                episode_steps = 0

                while state is not None:
                    state_t = _to_obs_tensor(state, device)
                    if state_t is None:
                        break

                    if rng.random() < epsilon:
                        env_action = float(rng.uniform(config.env.action_low, config.env.action_high))
                    else:
                        dist, _ = model.distribution(state_t)
                        raw_action = dist.loc if config.ppo.deterministic_eval else dist.sample()
                        squashed_action = torch.tanh(raw_action)
                        scaled_action = _scale_action(squashed_action, action_low, action_high)
                        env_action = _env_action_from_tensor(scaled_action.squeeze(0))

                    reward, done, _ = env.step(env_action)
                    reward_value = float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)
                    episode_return += reward_value
                    state = env.get_state()
                    episode_steps += 1
                    if done:
                        break

                returns.append(episode_return)
                if episode_steps == 0:
                    zero_step_episodes += 1
                cumulative_returns.append(getattr(env, "cumulative_return", []))
                base_env = _unwrap_base_env(env)
                return_rates.append(_compute_episode_return_rate(base_env))
    finally:
        if was_training:
            model.train()

    mean_return = float(np.mean(returns)) if returns else 0.0
    median_return = float(np.median(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    mean_return_rate = float(np.mean(return_rates)) if return_rates else 0.0
    std_return_rate = float(np.std(return_rates)) if return_rates else 0.0
    sharpe_ratio = float(mean_return_rate / std_return_rate) if std_return_rate > 1e-12 else 0.0
    win_rate = float(np.mean(np.array(returns) > 0.0)) if returns else 0.0

    metrics = {
        "mean_reward_return": mean_return,
        "median_reward_return": median_return,
        "std_reward_return": std_return,
        "mean_return_rate": mean_return_rate,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "initial_state_none_episodes": float(initial_state_none_episodes),
        "zero_step_episodes": float(zero_step_episodes),
        "episodes": float(episodes),
        "epsilon": float(epsilon),
    }
    return mean_return, metrics, cumulative_returns


def train(config: Config, run_paths: RunPaths) -> RunPaths:
    _validate_config(config)

    seed_everything(config.run.seed)
    device = _resolve_device(config.run.device)
    run_logger = setup_run_logger("train_ppo", run_paths.run_dir)
    run_logger.info("Logging dir: %s", run_paths.run_dir)

    df = _prepare_data(config)

    def _build_train_env():
        train_df, _ = sample_train_test_split(
            df,
            trading_period=config.env.trading_period,
            train_split=config.env.train_split,
        )
        return _build_env(config, train_df, device, trading_period=None)

    env = _build_train_env()
    obs_dim = int(getattr(env, "obs_dim", config.env.window_size))
    model = _build_model(config, obs_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    action_low, action_high = _action_bounds(config, device)

    log_paths = LogPaths(
        run_dir=run_paths.run_dir,
        metrics_csv=run_paths.metrics_csv,
        tensorboard_dir=run_paths.tensorboard_dir,
    )
    metrics_logger = MetricsLogger(
        log_paths,
        fieldnames=[
            "episode",
            "steps",
            "reward_return",
            "avg_policy_loss",
            "avg_value_loss",
            "avg_entropy",
            "avg_kl",
            "clip_fraction",
            "action_mean",
            "action_std",
            "avg_cash_ratio",
            "avg_position_ratio",
            "episode_return_rate",
        ],
    )

    save_config(config, run_paths.config_resolved)

    total_steps = 0
    stop_training = False

    for episode in range(config.train.num_episodes):
        if episode > 0 and config.train.resample_train_window_each_episode:
            env = _build_train_env()
            current_obs_dim = int(getattr(env, "obs_dim", config.env.window_size))
            if current_obs_dim != obs_dim:
                raise ValueError("Observation dimension changed after train-window resampling.")

        env.reset(start_index=env.window_size - 1)
        state = env.get_state()
        reward_return = 0.0
        steps = 0

        states: list[torch.Tensor] = []
        raw_actions: list[torch.Tensor] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []
        old_log_probs: list[float] = []
        episode_actions: list[float] = []
        cash_ratios: list[float] = []
        position_ratios: list[float] = []

        while state is not None:
            state_t = _to_obs_tensor(state, device)
            if state_t is None:
                break

            with torch.no_grad():
                dist, value_t = model.distribution(state_t)
                raw_action_t = dist.sample()
                squashed_action_t = torch.tanh(raw_action_t)
                scaled_action_t = _scale_action(squashed_action_t, action_low, action_high)
                log_prob_t = _squashed_log_prob(dist, raw_action_t, squashed_action_t)

            env_action = _env_action_from_tensor(scaled_action_t.squeeze(0))
            reward, done, _ = env.step(env_action)
            reward_value = float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)

            states.append(state_t.squeeze(0))
            raw_actions.append(raw_action_t.squeeze(0))
            rewards.append(reward_value)
            dones.append(float(done))
            values.append(float(value_t.item()))
            old_log_probs.append(float(log_prob_t.item()))
            episode_actions.append(env_action)

            base_env = _unwrap_base_env(env)
            if base_env is not None and hasattr(base_env, "get_account_features"):
                account_features = base_env.get_account_features()
                cash_ratios.append(float(account_features.get("cash_ratio", 0.0)))
                position_ratios.append(float(account_features.get("position_ratio", 0.0)))

            reward_return += reward_value
            steps += 1
            total_steps += 1
            state = env.get_state()

            if config.train.max_total_steps and total_steps >= config.train.max_total_steps:
                stop_training = True
                break
            if config.train.max_steps_per_episode and steps >= config.train.max_steps_per_episode:
                break
            if done:
                break

        if not states:
            row = {
                "episode": episode,
                "steps": steps,
                "reward_return": reward_return,
                "avg_policy_loss": 0.0,
                "avg_value_loss": 0.0,
                "avg_entropy": 0.0,
                "avg_kl": 0.0,
                "clip_fraction": 0.0,
                "action_mean": 0.0,
                "action_std": 0.0,
                "avg_cash_ratio": 0.0,
                "avg_position_ratio": 0.0,
                "episode_return_rate": 0.0,
            }
            metrics_logger.log(row, step=episode)
            if stop_training:
                break
            continue

        with torch.no_grad():
            if state is None:
                last_value = 0.0
            else:
                next_state_t = _to_obs_tensor(state, device)
                if next_state_t is None:
                    last_value = 0.0
                else:
                    _, last_value_t = model(next_state_t)
                    last_value = float(last_value_t.item())

        states_t = torch.stack(states, dim=0).to(device)
        raw_actions_t = torch.stack(raw_actions, dim=0).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        values_t = torch.tensor(values, dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

        advantages_t, returns_t = _compute_gae(
            rewards_t,
            values_t,
            dones_t,
            last_value=last_value,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )
        if config.ppo.normalize_advantage and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        batch_size = states_t.shape[0]
        minibatch_size = min(config.ppo.minibatch_size, batch_size)
        early_stop = False

        for _ in range(config.ppo.update_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_indices = permutation[start : start + minibatch_size]
                dist_mb, values_mb = model.distribution(states_t[mb_indices])
                raw_actions_mb = raw_actions_t[mb_indices]
                squashed_mb = torch.tanh(raw_actions_mb)
                new_log_probs = _squashed_log_prob(dist_mb, raw_actions_mb, squashed_mb)
                entropy = dist_mb.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_indices])
                mb_advantages = advantages_t[mb_indices]
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - config.ppo.clip_ratio, 1.0 + config.ppo.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(values_mb, returns_t[mb_indices])
                loss = policy_loss + config.ppo.value_coef * value_loss - config.ppo.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.ppo.max_grad_norm)
                optimizer.step()

                approx_kl = float((old_log_probs_t[mb_indices] - new_log_probs).mean().item())
                clip_frac = float(((ratio - 1.0).abs() > config.ppo.clip_ratio).float().mean().item())

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

                if config.ppo.target_kl is not None and approx_kl > config.ppo.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        base_env = _unwrap_base_env(env)
        episode_return_rate = _compute_episode_return_rate(base_env)
        row = {
            "episode": episode,
            "steps": steps,
            "reward_return": reward_return,
            "avg_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "avg_value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "avg_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "action_mean": float(np.mean(episode_actions)) if episode_actions else 0.0,
            "action_std": float(np.std(episode_actions)) if episode_actions else 0.0,
            "avg_cash_ratio": float(np.mean(cash_ratios)) if cash_ratios else 0.0,
            "avg_position_ratio": float(np.mean(position_ratios)) if position_ratios else 0.0,
            "episode_return_rate": float(episode_return_rate),
        }
        metrics_logger.log(row, step=episode)

        if (episode + 1) % config.train.log_interval == 0:
            run_logger.info(
                "Episode %s/%s | reward_return %.4f | return_rate %.2f%% | avg_policy_loss %.4f | avg_value_loss %.4f | avg_entropy %.4f | avg_kl %.5f",
                episode + 1,
                config.train.num_episodes,
                reward_return,
                row["episode_return_rate"] * 100.0,
                row["avg_policy_loss"],
                row["avg_value_loss"],
                row["avg_entropy"],
                row["avg_kl"],
            )

        if (episode + 1) % config.train.checkpoint_interval == 0:
            ckpt_path = run_paths.checkpoints_dir / f"episode_{episode + 1}.pt"
            save_checkpoint(
                ckpt_path,
                policy_state=model.state_dict(),
                target_state={},
                optimizer_state=optimizer.state_dict(),
                config=config_to_dict(config),
                episode=episode,
                step=total_steps,
            )

        if stop_training:
            break

    final_ckpt = run_paths.checkpoints_dir / "checkpoint_latest.pt"
    save_checkpoint(
        final_ckpt,
        policy_state=model.state_dict(),
        target_state={},
        optimizer_state=optimizer.state_dict(),
        config=config_to_dict(config),
        episode=config.train.num_episodes,
        step=total_steps,
    )
    metrics_logger.close()
    run_logger.info("Run artifacts saved to %s", run_paths.run_dir)
    return run_paths


def evaluate(
    config: Config,
    checkpoint_path: Path,
    episodes: int,
    epsilon: float,
    device: str,
    eval_seed: Optional[int] = None,
    eval_indices: Optional[List[int]] = None,
) -> Tuple[float, Dict[str, float], list]:
    _validate_config(config)
    resolved_device = _resolve_device(device)

    df = _prepare_data(config)
    dim_env = _build_env(config, df, resolved_device, trading_period=config.env.trading_period)
    obs_dim = int(getattr(dim_env, "obs_dim", config.env.window_size))
    model = _build_model(config, obs_dim, resolved_device)

    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    model.load_state_dict(checkpoint["policy_state"])

    return _evaluate_with_model(
        config,
        model,
        episodes=episodes,
        epsilon=epsilon,
        device=resolved_device,
        eval_seed=eval_seed,
        eval_indices=eval_indices,
        eval_df=df,
    )
