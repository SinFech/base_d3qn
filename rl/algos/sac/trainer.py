from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from rl.algos.sac.networks import SACGaussianPolicy, SACQNetwork
from rl.algos.sac.replay_buffer import ReplayBuffer
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
    # Keep legacy discrete fields for compatibility.
    max_positions: Optional[int] = None
    sell_mode: str = "all"
    obs: ObsConfig = field(default_factory=ObsConfig)


@dataclass
class SACConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    replay_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 0.0003
    critic_learning_rate: float = 0.0003
    alpha_learning_rate: float = 0.0003
    init_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy_scale: float = 1.0
    start_steps: int = 1_000
    update_after: int = 1_000
    update_every: int = 1
    updates_per_step: int = 1
    action_dim: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    deterministic_eval: bool = True


@dataclass
class TrainConfig:
    num_episodes: int = 100
    max_steps_per_episode: Optional[int] = None
    max_total_steps: Optional[int] = None
    log_interval: int = 10
    checkpoint_interval: int = 5
    resample_train_window_each_episode: bool = False


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
    sac: SACConfig
    train: TrainConfig
    eval: EvalConfig
    run: RunConfig
    algo: str = "sac_continuous"


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


def _legacy_sac_payload(raw: Dict) -> Dict:
    payload: Dict = {}
    raw_agent = raw.get("agent", {}) or {}
    raw_model = raw.get("model", {}) or {}
    if "learning_rate" in raw_agent:
        payload["actor_learning_rate"] = raw_agent["learning_rate"]
        payload["critic_learning_rate"] = raw_agent["learning_rate"]
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
        account_features=raw_signature.get("account_features", []),
        logsig=LogSigConfig(**raw_logsig),
        torch=SignatureTorchConfig(**raw_signature_torch),
        perf=SignaturePerfConfig(**raw_signature_perf),
    )
    obs_cfg = ObsConfig(type=raw_obs.get("type", "raw"), signature=signature_cfg)
    env_payload = {key: value for key, value in raw_env.items() if key != "obs"}

    merged_sac = _legacy_sac_payload(data)
    merged_sac.update(data.get("sac", {}) or {})
    config = Config(
        data=DataConfig(**data.get("data", {})),
        env=EnvConfig(**env_payload, obs=obs_cfg),
        sac=SACConfig(**merged_sac),
        train=TrainConfig(**data.get("train", {})),
        eval=EvalConfig(**data.get("eval", {})),
        run=RunConfig(**data.get("run", {})),
        algo=str(data.get("algo", "sac_continuous")),
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
        raise ValueError("SAC trainer requires env.action_mode='continuous'.")
    if config.sac.action_dim != 1:
        raise ValueError("Current continuous trading environment expects sac.action_dim == 1.")
    if config.env.action_high <= config.env.action_low:
        raise ValueError("env.action_high must be greater than env.action_low.")
    if (not config.env.allow_short) and config.env.action_low < 0:
        raise ValueError("env.action_low must be >= 0 when allow_short is false.")
    if config.env.initial_capital <= 0:
        raise ValueError("env.initial_capital must be > 0.")
    if config.sac.batch_size < 1:
        raise ValueError("sac.batch_size must be >= 1.")
    if config.sac.replay_size < config.sac.batch_size:
        raise ValueError("sac.replay_size must be >= sac.batch_size.")


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


def _build_networks(config: Config, obs_dim: int, device: str):
    actor = SACGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=config.sac.action_dim,
        hidden_sizes=config.sac.hidden_sizes,
        action_low=config.env.action_low,
        action_high=config.env.action_high,
        log_std_min=config.sac.log_std_min,
        log_std_max=config.sac.log_std_max,
    ).to(device)
    q1 = SACQNetwork(obs_dim, config.sac.action_dim, config.sac.hidden_sizes).to(device)
    q2 = SACQNetwork(obs_dim, config.sac.action_dim, config.sac.hidden_sizes).to(device)
    target_q1 = copy.deepcopy(q1).to(device)
    target_q2 = copy.deepcopy(q2).to(device)
    target_q1.eval()
    target_q2.eval()
    return actor, q1, q2, target_q1, target_q2


def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)


def _to_obs_tensor(obs, device: str) -> Optional[torch.Tensor]:
    if obs is None:
        return None
    if isinstance(obs, torch.Tensor):
        out = obs.to(device=device, dtype=torch.float32)
    else:
        out = torch.as_tensor(obs, device=device, dtype=torch.float32)
    return out.reshape(1, -1)


def _to_obs_numpy(obs, obs_dim: int) -> np.ndarray:
    if obs is None:
        return np.zeros(obs_dim, dtype=np.float32)
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] != obs_dim:
        raise ValueError(f"obs dimension mismatch: expected {obs_dim}, got {arr.shape[0]}")
    return arr


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


def _sample_env_action_from_actor(actor: SACGaussianPolicy, state_t: torch.Tensor) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        action_t, _ = actor.sample(state_t)
    action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return action_np, float(action_np[0])


def _deterministic_env_action_from_actor(
    actor: SACGaussianPolicy,
    state_t: torch.Tensor,
) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        action_t = actor.deterministic(state_t)
    action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return action_np, float(action_np[0])


def _train_step(
    config: Config,
    actor: SACGaussianPolicy,
    q1: SACQNetwork,
    q2: SACQNetwork,
    target_q1: SACQNetwork,
    target_q2: SACQNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    alpha: torch.Tensor,
    log_alpha: Optional[torch.Tensor],
    alpha_optimizer: Optional[optim.Optimizer],
    target_entropy: Optional[float],
) -> dict[str, float]:
    batch = replay_buffer.sample(config.sac.batch_size)
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]

    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_obs)
        target_q1_values = target_q1(next_obs, next_actions)
        target_q2_values = target_q2(next_obs, next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values) - alpha * next_log_probs
        td_target = rewards + config.sac.gamma * (1.0 - dones) * target_q_values

    q1_loss = F.mse_loss(q1(obs, actions), td_target)
    q2_loss = F.mse_loss(q2(obs, actions), td_target)
    q_loss = q1_loss + q2_loss

    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    new_actions, log_probs = actor.sample(obs)
    q_new_actions = torch.min(q1(obs, new_actions), q2(obs, new_actions))
    actor_loss = (alpha * log_probs - q_new_actions).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss_value = 0.0
    if config.sac.auto_alpha:
        if log_alpha is None or alpha_optimizer is None or target_entropy is None:
            raise ValueError("Auto alpha enabled but alpha optimizer state is missing.")
        alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha = log_alpha.exp().detach()
        alpha_loss_value = float(alpha_loss.item())

    _soft_update(q1, target_q1, config.sac.tau)
    _soft_update(q2, target_q2, config.sac.tau)

    entropy = float((-log_probs).mean().item())
    return {
        "q1_loss": float(q1_loss.item()),
        "q2_loss": float(q2_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": alpha_loss_value,
        "alpha": float(alpha.item()),
        "entropy": entropy,
    }


def _evaluate_with_actor(
    config: Config,
    actor: SACGaussianPolicy,
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
    returns: List[float] = []
    return_rates: List[float] = []
    cumulative_returns: List[list] = []
    initial_state_none_episodes = 0
    zero_step_episodes = 0

    was_training = actor.training
    actor.eval()
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
                        if config.sac.deterministic_eval:
                            _, env_action = _deterministic_env_action_from_actor(actor, state_t)
                        else:
                            _, env_action = _sample_env_action_from_actor(actor, state_t)

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
            actor.train()

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
    run_logger = setup_run_logger("train_sac", run_paths.run_dir)
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
    actor, q1, q2, target_q1, target_q2 = _build_networks(config, obs_dim, device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.sac.actor_learning_rate)
    q_optimizer = optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=config.sac.critic_learning_rate,
    )

    if config.sac.auto_alpha:
        log_alpha = torch.tensor(np.log(config.sac.init_alpha), device=device, requires_grad=True)
        alpha_optimizer = optim.Adam([log_alpha], lr=config.sac.alpha_learning_rate)
        alpha = log_alpha.exp().detach()
        target_entropy = -float(config.sac.action_dim) * float(config.sac.target_entropy_scale)
    else:
        log_alpha = None
        alpha_optimizer = None
        alpha = torch.tensor(float(config.sac.init_alpha), device=device)
        target_entropy = None

    replay_buffer = ReplayBuffer(config.sac.replay_size, obs_dim, config.sac.action_dim, device=device)

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
            "episode_return_rate",
            "avg_q1_loss",
            "avg_q2_loss",
            "avg_actor_loss",
            "avg_alpha_loss",
            "alpha",
            "avg_entropy",
            "action_mean",
            "action_std",
            "avg_cash_ratio",
            "avg_position_ratio",
        ],
    )

    save_config(config, run_paths.config_resolved)

    total_steps = 0
    stop_training = False
    rng = np.random.default_rng(config.run.seed)

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

        q1_losses: list[float] = []
        q2_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        entropies: list[float] = []
        episode_actions: list[float] = []
        cash_ratios: list[float] = []
        position_ratios: list[float] = []

        while state is not None:
            state_np = _to_obs_numpy(state, obs_dim)
            state_t = _to_obs_tensor(state_np, device)
            if state_t is None:
                break

            if total_steps < config.sac.start_steps:
                action_np = np.array(
                    [rng.uniform(config.env.action_low, config.env.action_high)],
                    dtype=np.float32,
                )
                env_action = float(action_np[0])
            else:
                action_np, env_action = _sample_env_action_from_actor(actor, state_t)

            reward, done, _ = env.step(env_action)
            reward_value = float(reward.item()) if isinstance(reward, torch.Tensor) else float(reward)
            next_state = env.get_state()
            next_state_np = _to_obs_numpy(next_state, obs_dim)
            done_float = 1.0 if done else 0.0
            replay_buffer.add(state_np, action_np, reward_value, next_state_np, done_float)

            state = next_state
            reward_return += reward_value
            steps += 1
            total_steps += 1
            episode_actions.append(env_action)

            base_env = _unwrap_base_env(env)
            if base_env is not None and hasattr(base_env, "get_account_features"):
                account_features = base_env.get_account_features()
                cash_ratios.append(float(account_features.get("cash_ratio", 0.0)))
                position_ratios.append(float(account_features.get("position_ratio", 0.0)))

            if total_steps >= config.sac.update_after and len(replay_buffer) >= config.sac.batch_size:
                if total_steps % config.sac.update_every == 0:
                    for _ in range(config.sac.updates_per_step * config.sac.update_every):
                        update_metrics = _train_step(
                            config=config,
                            actor=actor,
                            q1=q1,
                            q2=q2,
                            target_q1=target_q1,
                            target_q2=target_q2,
                            actor_optimizer=actor_optimizer,
                            q_optimizer=q_optimizer,
                            replay_buffer=replay_buffer,
                            alpha=alpha,
                            log_alpha=log_alpha,
                            alpha_optimizer=alpha_optimizer,
                            target_entropy=target_entropy,
                        )
                        if config.sac.auto_alpha and log_alpha is not None:
                            alpha = log_alpha.exp().detach()
                        q1_losses.append(update_metrics["q1_loss"])
                        q2_losses.append(update_metrics["q2_loss"])
                        actor_losses.append(update_metrics["actor_loss"])
                        alpha_losses.append(update_metrics["alpha_loss"])
                        entropies.append(update_metrics["entropy"])

            if config.train.max_total_steps and total_steps >= config.train.max_total_steps:
                stop_training = True
                break
            if config.train.max_steps_per_episode and steps >= config.train.max_steps_per_episode:
                break
            if done:
                break

        base_env = _unwrap_base_env(env)
        episode_return_rate = _compute_episode_return_rate(base_env)
        row = {
            "episode": episode,
            "steps": steps,
            "reward_return": reward_return,
            "episode_return_rate": float(episode_return_rate),
            "avg_q1_loss": float(np.mean(q1_losses)) if q1_losses else 0.0,
            "avg_q2_loss": float(np.mean(q2_losses)) if q2_losses else 0.0,
            "avg_actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "avg_alpha_loss": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
            "alpha": float(alpha.item()),
            "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "action_mean": float(np.mean(episode_actions)) if episode_actions else 0.0,
            "action_std": float(np.std(episode_actions)) if episode_actions else 0.0,
            "avg_cash_ratio": float(np.mean(cash_ratios)) if cash_ratios else 0.0,
            "avg_position_ratio": float(np.mean(position_ratios)) if position_ratios else 0.0,
        }
        metrics_logger.log(row, step=episode)

        if (episode + 1) % config.train.log_interval == 0:
            run_logger.info(
                "Episode %s/%s | reward_return %.4f | return_rate %.2f%% | q1_loss %.4f | q2_loss %.4f | actor_loss %.4f | alpha %.4f",
                episode + 1,
                config.train.num_episodes,
                reward_return,
                row["episode_return_rate"] * 100.0,
                row["avg_q1_loss"],
                row["avg_q2_loss"],
                row["avg_actor_loss"],
                row["alpha"],
            )

        if (episode + 1) % config.train.checkpoint_interval == 0:
            ckpt_path = run_paths.checkpoints_dir / f"episode_{episode + 1}.pt"
            save_checkpoint(
                ckpt_path,
                policy_state={
                    "actor": actor.state_dict(),
                    "q1": q1.state_dict(),
                    "q2": q2.state_dict(),
                    "target_q1": target_q1.state_dict(),
                    "target_q2": target_q2.state_dict(),
                    "log_alpha": float(log_alpha.item()) if log_alpha is not None else None,
                    "alpha": float(alpha.item()),
                },
                target_state={},
                optimizer_state={
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
                },
                config=config_to_dict(config),
                episode=episode,
                step=total_steps,
            )

        if stop_training:
            break

    final_ckpt = run_paths.checkpoints_dir / "checkpoint_latest.pt"
    save_checkpoint(
        final_ckpt,
        policy_state={
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "log_alpha": float(log_alpha.item()) if log_alpha is not None else None,
            "alpha": float(alpha.item()),
        },
        target_state={},
        optimizer_state={
            "actor_optimizer": actor_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "alpha_optimizer": alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
        },
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
    actor, _, _, _, _ = _build_networks(config, obs_dim, resolved_device)

    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    payload = checkpoint.get("policy_state", {})
    if "actor" in payload:
        actor.load_state_dict(payload["actor"])
    else:
        actor.load_state_dict(payload)

    return _evaluate_with_actor(
        config=config,
        actor=actor,
        episodes=episodes,
        epsilon=epsilon,
        device=resolved_device,
        eval_seed=eval_seed,
        eval_indices=eval_indices,
        eval_df=df,
    )
