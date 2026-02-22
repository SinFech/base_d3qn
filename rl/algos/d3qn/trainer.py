from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from rl.algos.d3qn.agent import D3QNAgent
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
class EnvConfig:
    reward: str = "profit"
    window_size: int = 24
    trading_period: int = 500
    train_split: float = 0.8
    max_positions: Optional[int] = None
    sell_mode: str = "all"
    obs: "ObsConfig" = field(default_factory=lambda: ObsConfig())


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
    logsig: LogSigConfig = field(default_factory=LogSigConfig)
    torch: SignatureTorchConfig = field(default_factory=SignatureTorchConfig)
    perf: SignaturePerfConfig = field(default_factory=SignaturePerfConfig)


@dataclass
class ObsConfig:
    type: str = "raw"
    signature: SignatureObsConfig = field(default_factory=SignatureObsConfig)


@dataclass
class AgentConfig:
    replay_mem_size: int = 10000
    batch_size: int = 40
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_steps: int = 200
    learning_rate: float = 0.0005
    input_dim: int = 24
    hidden_dim: int = 120
    action_number: int = 3
    target_update: int = 5
    model: str = "ddqn"
    double: bool = True


@dataclass
class TrainConfig:
    num_episodes: int = 100
    max_steps_per_episode: Optional[int] = None
    max_total_steps: Optional[int] = None
    log_interval: int = 10
    checkpoint_interval: int = 5
    eval_epsilon: float = 0.0
    resample_train_window_each_episode: bool = False
    eval_interval: int = 10
    eval_episodes: int = 20
    eval_seed: Optional[int] = 20240101
    plateau_threshold: float = 0.01
    plateau_patience: int = 5


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
class ModelConfig:
    type: str = "conv_dueling"
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class Config:
    data: DataConfig
    env: EnvConfig
    agent: AgentConfig
    train: TrainConfig
    eval: EvalConfig
    run: RunConfig
    model: ModelConfig


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


def _compute_fixed_eval_indices(
    data_len: int,
    window_size: int,
    trading_period: Optional[int],
    num_episodes: int,
    seed: int,
) -> List[int]:
    """Precompute fixed start indices for periodic eval using a local RNG."""
    min_start, max_start = _compute_start_range(data_len, window_size, trading_period)
    if max_start < min_start:
        raise ValueError(
            f"Invalid start_index range [{min_start}, {max_start}] for data length {data_len} "
            f"and trading_period {trading_period}."
        )
    rng = np.random.default_rng(seed)
    return rng.integers(min_start, max_start + 1, size=num_episodes).tolist()


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
        logsig=LogSigConfig(**raw_logsig),
        torch=SignatureTorchConfig(**raw_signature_torch),
        perf=SignaturePerfConfig(**raw_signature_perf),
    )
    obs_cfg = ObsConfig(type=raw_obs.get("type", "raw"), signature=signature_cfg)

    agent_cfg = AgentConfig(**data.get("agent", {}))
    raw_model = data.get("model", {})
    if raw_model:
        model_cfg = ModelConfig(**raw_model)
    else:
        model_map = {"ddqn": "conv_dueling", "dqn": "conv"}
        model_cfg = ModelConfig(type=model_map.get(agent_cfg.model, "conv_dueling"))

    env_payload = {key: value for key, value in raw_env.items() if key != "obs"}
    return Config(
        data=DataConfig(**data.get("data", {})),
        env=EnvConfig(**env_payload, obs=obs_cfg),
        agent=agent_cfg,
        train=TrainConfig(**data.get("train", {})),
        eval=EvalConfig(**data.get("eval", {})),
        run=RunConfig(**data.get("run", {})),
        model=model_cfg,
    )


def config_to_dict(config: Config) -> Dict:
    return asdict(config)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    return config_from_dict(raw)


def save_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))


def build_agent(config: Config, device: str, input_dim: Optional[int] = None) -> D3QNAgent:
    resolved_input_dim = input_dim if input_dim is not None else config.agent.input_dim
    return D3QNAgent(
        replay_mem_size=config.agent.replay_mem_size,
        batch_size=config.agent.batch_size,
        gamma=config.agent.gamma,
        eps_start=config.agent.eps_start,
        eps_end=config.agent.eps_end,
        eps_steps=config.agent.eps_steps,
        learning_rate=config.agent.learning_rate,
        input_dim=resolved_input_dim,
        hidden_dim=config.agent.hidden_dim,
        action_number=config.agent.action_number,
        target_update=config.agent.target_update,
        model=config.model.type,
        hidden_sizes=config.model.hidden_sizes,
        double=config.agent.double,
        device=device,
    )


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


def _resolve_obs_dim(env, config: Config) -> int:
    obs_dim = getattr(env, "obs_dim", config.env.window_size)
    if config.env.obs.type == "raw" and config.agent.input_dim != obs_dim:
        raise ValueError("agent.input_dim must match env.window_size for raw observations")
    config.agent.input_dim = obs_dim
    return obs_dim


def _unwrap_trading_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "agent_positions"):
            return current
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return None


def _evaluate_with_agent(
    config: Config,
    agent: D3QNAgent,
    episodes: int,
    epsilon: float,
    device: str,
    eval_seed: Optional[int] = None,
    eval_indices: Optional[List[int]] = None,
    eval_df=None,
) -> Tuple[float, Dict[str, float], list]:
    seed = eval_seed if eval_seed is not None else config.run.seed
    seed_everything(seed)

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
        for index in start_indices:
            if not (min_start <= index <= max_start):
                raise ValueError(
                    "eval index must be in the range "
                    f"[{min_start}, {max_start}] but got {index}."
                )
    else:
        rng = np.random.default_rng(seed)
        start_indices = rng.integers(min_start, max_start + 1, size=episodes).tolist()

    env = make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        trading_period=config.env.trading_period,
        max_positions=config.env.max_positions,
        sell_mode=config.env.sell_mode,
        obs_config=config.env.obs,
    )
    _resolve_obs_dim(env, config)

    returns: List[float] = []
    return_rates: List[float] = []
    cumulative_returns: List[list] = []
    initial_state_none_episodes = 0
    zero_step_episodes = 0

    policy_was_training = agent.policy_net.training
    target_was_training = agent.target_net.training
    agent.policy_net.eval()
    agent.target_net.eval()
    try:
        with torch.no_grad():
            for start_index in start_indices:
                env.reset(start_index=start_index)
                agent.reset_episode()
                state = env.get_state()
                if state is None:
                    initial_state_none_episodes += 1
                    returns.append(0.0)
                    cumulative_returns.append(getattr(env, "cumulative_return", []))
                    return_rates.append(0.0)
                    continue

                episode_return = 0.0
                episode_steps = 0
                while state is not None:
                    action = agent.select_action(state, training=False, epsilon_override=epsilon)
                    reward, done, _ = env.step(action)
                    episode_return += reward.item()
                    state = env.get_state()
                    episode_steps += 1
                    if done:
                        break

                returns.append(episode_return)
                if episode_steps == 0:
                    zero_step_episodes += 1
                cumulative_returns.append(env.cumulative_return)
                episode_return_rate = 0.0
                base_env = _unwrap_trading_env(env)
                if (
                    base_env is not None
                    and hasattr(base_env, "init_price")
                    and hasattr(base_env, "realized_pnl")
                    and hasattr(base_env, "agent_open_position_value")
                ):
                    equity_start = float(base_env.init_price)
                    equity_end = equity_start + float(base_env.realized_pnl) + float(base_env.agent_open_position_value)
                    episode_return_rate = (equity_end / (equity_start + 1e-8)) - 1.0
                return_rates.append(episode_return_rate)
    finally:
        if policy_was_training:
            agent.policy_net.train()
        if target_was_training:
            agent.target_net.train()

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

    seed_everything(config.run.seed)
    device = _resolve_device(config.run.device)
    run_logger = setup_run_logger("train", run_paths.run_dir)
    run_logger.info("Logging dir: %s", run_paths.run_dir)

    df = _prepare_data(config)

    def _build_train_env():
        train_df, _ = sample_train_test_split(
            df,
            trading_period=config.env.trading_period,
            train_split=config.env.train_split,
        )
        return make_env(
            train_df,
            config.env.reward,
            config.env.window_size,
            device,
            max_positions=config.env.max_positions,
            sell_mode=config.env.sell_mode,
            obs_config=config.env.obs,
        )

    env = _build_train_env()
    obs_dim = _resolve_obs_dim(env, config)
    agent = build_agent(config, device, input_dim=obs_dim)

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
            "epsilon",
            "avg_loss",
            "avg_q",
        ],
    )

    save_config(config, run_paths.config_resolved)

    total_steps = 0
    stop_training = False
    eval_history: List[Dict[str, float]] = []
    eval_seed = getattr(config.train, "eval_seed", None) or config.eval.seed
    eval_interval = getattr(config.train, "eval_interval", 10)
    eval_episodes = getattr(config.train, "eval_episodes", 20)

    fixed_eval_indices: Optional[List[int]] = None
    if eval_interval > 0 and eval_episodes > 0:
        fixed_eval_indices = _compute_fixed_eval_indices(
            len(df),
            config.env.window_size,
            config.env.trading_period,
            eval_episodes,
            eval_seed,
        )

    for episode in range(config.train.num_episodes):
        if episode > 0 and config.train.resample_train_window_each_episode:
            env = _build_train_env()
            current_obs_dim = getattr(env, "obs_dim", config.env.window_size)
            if current_obs_dim != obs_dim:
                raise ValueError("Observation dimension changed after train-window resampling.")
        env.reset(start_index=env.window_size - 1)
        agent.reset_episode()
        state = env.get_state()
        reward_return = 0.0
        losses = []
        q_values = []
        steps = 0

        while state is not None:
            action = agent.select_action(state, training=True)
            reward, done, _ = env.step(action)
            reward_return += reward.item()
            next_state = env.get_state()
            agent.store_transition(state, action, next_state, reward)

            if agent.double:
                result = agent.optimize_double_dqn()
            else:
                result = agent.optimize()
            if result:
                loss_value, q_value = result
                losses.append(loss_value)
                q_values.append(q_value)

            state = next_state
            steps += 1
            total_steps += 1
            if config.train.max_total_steps and total_steps >= config.train.max_total_steps:
                stop_training = True
                break

            if config.train.max_steps_per_episode and steps >= config.train.max_steps_per_episode:
                break
            if done:
                break

        if episode % agent.target_update == 0:
            agent.update_target()

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_q = float(np.mean(q_values)) if q_values else 0.0
        row = {
            "episode": episode,
            "steps": steps,
            "reward_return": reward_return,
            "epsilon": agent.last_epsilon,
            "avg_loss": avg_loss,
            "avg_q": avg_q,
        }
        metrics_logger.log(row, step=episode)

        if (episode + 1) % config.train.log_interval == 0:
            run_logger.info(
                "Episode %s/%s | reward_return %.2f | epsilon %.3f | avg_loss %.4f | avg_q %.4f",
                episode + 1,
                config.train.num_episodes,
                reward_return,
                agent.last_epsilon,
                avg_loss,
                avg_q,
            )

        if (episode + 1) % config.train.checkpoint_interval == 0:
            ckpt_path = run_paths.checkpoints_dir / f"episode_{episode + 1}.pt"
            save_checkpoint(
                ckpt_path,
                policy_state=agent.policy_net.state_dict(),
                target_state=agent.target_net.state_dict(),
                optimizer_state=agent.optimizer.state_dict(),
                config=config_to_dict(config),
                episode=episode,
                step=total_steps,
            )

        if eval_interval > 0 and eval_episodes > 0 and (episode + 1) % eval_interval == 0:
            _, eval_metrics, _ = _evaluate_with_agent(
                config,
                agent,
                episodes=eval_episodes,
                epsilon=0.0,
                device=device,
                eval_seed=eval_seed,
                eval_indices=fixed_eval_indices,
                eval_df=df,
            )
            row_eval = {
                "eval_at_episode": episode + 1,
                "mean_reward_return": eval_metrics["mean_reward_return"],
                "median_reward_return": eval_metrics["median_reward_return"],
                "std_reward_return": eval_metrics["std_reward_return"],
                # Store return rate as percentage for easier reading in CSV.
                "mean_return_rate": eval_metrics["mean_return_rate"] * 100.0,
                "sharpe_ratio": eval_metrics["sharpe_ratio"],
                "win_rate": eval_metrics["win_rate"],
            }
            eval_history.append(row_eval)
            run_logger.info(
                "Eval at episode %s | mean_reward_return %.2f | mean_return_rate %.2f%% | sharpe_ratio %.4f | win_rate %.2f",
                row_eval["eval_at_episode"],
                row_eval["mean_reward_return"],
                row_eval["mean_return_rate"],
                row_eval["sharpe_ratio"],
                row_eval["win_rate"],
            )
            if eval_metrics.get("initial_state_none_episodes", 0) > 0 or eval_metrics.get("zero_step_episodes", 0) > 0:
                run_logger.warning(
                    "Eval diagnostics at episode %s: initial_state_none=%s, zero_step_episodes=%s",
                    row_eval["eval_at_episode"],
                    int(eval_metrics.get("initial_state_none_episodes", 0)),
                    int(eval_metrics.get("zero_step_episodes", 0)),
                )

        if stop_training:
            break

    final_ckpt = run_paths.checkpoints_dir / "checkpoint_latest.pt"
    save_checkpoint(
        final_ckpt,
        policy_state=agent.policy_net.state_dict(),
        target_state=agent.target_net.state_dict(),
        optimizer_state=agent.optimizer.state_dict(),
        config=config_to_dict(config),
        episode=config.train.num_episodes,
        step=total_steps,
    )
    if eval_history:
        eval_csv = run_paths.run_dir / "eval_history.csv"
        with eval_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "eval_at_episode",
                    "mean_reward_return",
                    "median_reward_return",
                    "std_reward_return",
                    "mean_return_rate",
                    "sharpe_ratio",
                    "win_rate",
                ],
            )
            w.writeheader()
            w.writerows(eval_history)
        run_logger.info("Eval history written to %s", eval_csv)
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
    device = _resolve_device(device)

    df = _prepare_data(config)
    dim_env = make_env(
        df,
        config.env.reward,
        config.env.window_size,
        device,
        trading_period=config.env.trading_period,
        max_positions=config.env.max_positions,
        sell_mode=config.env.sell_mode,
        obs_config=config.env.obs,
    )
    obs_dim = _resolve_obs_dim(dim_env, config)
    agent = build_agent(config, device, input_dim=obs_dim)

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.target_net.load_state_dict(checkpoint["target_state"])
    return _evaluate_with_agent(
        config,
        agent,
        episodes=episodes,
        epsilon=epsilon,
        device=device,
        eval_seed=eval_seed,
        eval_indices=eval_indices,
        eval_df=df,
    )
