from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def print_stats(model: str, c_return, table) -> None:
    c_return = np.array(c_return).flatten()
    table.add_row(
        [
            str(model),
            f"{np.mean(c_return):.2f}",
            f"{np.amax(c_return):.2f}",
            f"{np.amin(c_return):.2f}",
            f"{np.std(c_return):.2f}",
        ]
    )


def plot_conf_interval(name: str, cum_returns) -> None:
    """cum_returns must be 2D."""
    mean = np.mean(np.array(cum_returns), axis=0)
    std = np.std(np.array(cum_returns), axis=0)
    lower = mean - 0.95 * std
    upper = mean + 0.95 * std

    plt.figure(figsize=(20, 5))
    plt.xlabel("Trading Instant (h)")
    plt.ylabel(name)
    plt.legend(["Cumulative Average Return (%)"], loc="upper left")
    plt.grid(True)
    plt.ylim(-5, 15)
    plt.plot(range(len(mean)), mean, linewidth=2)
    plt.fill_between(range(len(mean)), lower, upper, color="b", alpha=0.2)
    plt.show()


def plot_multiple_conf_interval(names, cum_returns_list) -> None:
    """cum_returns_list[i] must be 2D."""
    for idx, returns in enumerate(cum_returns_list, start=1):
        plt.subplot(len(cum_returns_list), 2, idx)
        mean = np.mean(np.array(returns), axis=0)
        std = np.std(np.array(returns), axis=0)
        lower = mean - 0.95 * std
        upper = mean + 0.95 * std

        plt.xlabel("Trading Instant (h)")
        plt.ylabel(names[idx - 1])
        plt.title("Cumulative Average Return (%)")
        plt.grid(True)
        plt.plot(range(len(mean)), mean, linewidth=2)
        plt.fill_between(range(len(mean)), lower, upper, color="b", alpha=0.2)

    plt.show()


def plot_actions(action_history: List[tuple]) -> None:
    states = [entry[0] for entry in action_history]
    actions = [entry[1] for entry in action_history]

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(states)), states, c=actions, cmap="viridis", label="Action by State")
    plt.colorbar(label="Action")
    plt.xlabel("Step")
    plt.ylabel("State Value")
    plt.title("Action Taken at Each State")
    plt.legend()
    plt.show()
