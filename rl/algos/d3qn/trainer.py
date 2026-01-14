from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from rl.algos.d3qn.agent import D3QNAgent
from rl.envs.make_env import filter_date_range, load_price_data, make_env, sample_train_test_split
from rl.utils.checkpoint import load_checkpoint, save_checkpoint
from rl.utils.logging import LogPaths, MetricsLogger
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


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    data: DataConfig
    env: EnvConfig
    agent: AgentConfig
    train: TrainConfig
    run: RunConfig


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def config_from_dict(data: Dict) -> Config:
    return Config(
        data=DataConfig(**data.get("data", {})),
        env=EnvConfig(**data.get("env", {})),
        agent=AgentConfig(**data.get("agent", {})),
        train=TrainConfig(**data.get("train", {})),
        run=RunConfig(**data.get("run", {})),
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


def build_agent(config: Config, device: str) -> D3QNAgent:
    return D3QNAgent(
        replay_mem_size=config.agent.replay_mem_size,
        batch_size=config.agent.batch_size,
        gamma=config.agent.gamma,
        eps_start=config.agent.eps_start,
        eps_end=config.agent.eps_end,
        eps_steps=config.agent.eps_steps,
        learning_rate=config.agent.learning_rate,
        input_dim=config.agent.input_dim,
        hidden_dim=config.agent.hidden_dim,
        action_number=config.agent.action_number,
        target_update=config.agent.target_update,
        model=config.agent.model,
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


def train(config: Config, run_paths: RunPaths) -> RunPaths:
    if config.agent.input_dim != config.env.window_size:
        raise ValueError("agent.input_dim must match env.window_size")

    seed_everything(config.run.seed)
    device = _resolve_device(config.run.device)

    df = _prepare_data(config)
    train_df, _ = sample_train_test_split(
        df,
        trading_period=config.env.trading_period,
        train_split=config.env.train_split,
    )

    env = make_env(train_df, config.env.reward, config.env.window_size, device)
    agent = build_agent(config, device)

    log_paths = LogPaths(
        run_dir=run_paths.run_dir,
        metrics_csv=run_paths.metrics_csv,
        tensorboard_dir=run_paths.tensorboard_dir,
    )
    logger = MetricsLogger(
        log_paths,
        fieldnames=[
            "episode",
            "steps",
            "episode_return",
            "epsilon",
            "avg_loss",
            "avg_q",
        ],
    )

    save_config(config, run_paths.config_resolved)

    total_steps = 0
    stop_training = False
    for episode in range(config.train.num_episodes):
        env.reset()
        agent.reset_episode()
        state = env.get_state()
        episode_return = 0.0
        losses = []
        q_values = []
        steps = 0

        while state is not None:
            action = agent.select_action(state, training=True)
            reward, done, _ = env.step(action)
            episode_return += reward.item()
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
            "episode_return": episode_return,
            "epsilon": agent.last_epsilon,
            "avg_loss": avg_loss,
            "avg_q": avg_q,
        }
        logger.log(row, step=episode)

        if (episode + 1) % config.train.log_interval == 0:
            print(
                f"Episode {episode + 1}/{config.train.num_episodes} | "
                f"return {episode_return:.2f} | "
                f"epsilon {agent.last_epsilon:.3f} | "
                f"avg_loss {avg_loss:.4f} | "
                f"avg_q {avg_q:.4f}"
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
    logger.close()
    return run_paths


def evaluate(
    config: Config,
    checkpoint_path: Path,
    episodes: int,
    epsilon: float,
    device: str,
) -> Tuple[float, Dict[str, float], list[list[float]]]:
    device = _resolve_device(device)
    seed_everything(config.run.seed)

    df = _prepare_data(config)
    agent = build_agent(config, device)

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.target_net.load_state_dict(checkpoint["target_state"])
    agent.policy_net.eval()

    returns = []
    cumulative_returns = []
    for _ in range(episodes):
        _, test_df = sample_train_test_split(
            df,
            trading_period=config.env.trading_period,
            train_split=config.env.train_split,
        )
        env = make_env(test_df, config.env.reward, config.env.window_size, device)
        env.reset()
        state = env.get_state()
        episode_return = 0.0

        while state is not None:
            action = agent.select_action(state, training=False, epsilon_override=epsilon)
            reward, done, _ = env.step(action)
            episode_return += reward.item()
            state = env.get_state()
            if done:
                break
        returns.append(episode_return)
        cumulative_returns.append(env.cumulative_return)

    mean_return = float(np.mean(returns)) if returns else 0.0
    metrics = {
        "mean_return": mean_return,
        "episodes": float(episodes),
        "epsilon": float(epsilon),
    }
    return mean_return, metrics, cumulative_returns
