from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.optim as optim

from rl.algos.d3qn.losses import mse_q_loss
from rl.algos.d3qn.networks import build_q_network
from rl.algos.d3qn.replay_buffer import ReplayBuffer, Transition
from rl.algos.d3qn.schedules import EpsilonSchedule


class D3QNAgent:
    def __init__(
        self,
        replay_mem_size: int = 10000,
        batch_size: int = 40,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_steps: int = 200,
        learning_rate: float = 0.0005,
        input_dim: int = 24,
        hidden_dim: int = 120,
        action_number: int = 3,
        target_update: int = 5,
        model: str = "ddqn",
        double: bool = True,
        device: str = "auto",
    ) -> None:
        self.replay_mem_size = replay_mem_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_number = action_number
        self.target_update = target_update
        self.model = model
        self.double = double
        self.training = True

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.policy_net = build_q_network(self.model, self.input_dim, self.action_number).to(self.device)
        self.target_net = build_q_network(self.model, self.input_dim, self.action_number).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.replay_mem_size)
        self.epsilon_schedule = EpsilonSchedule(self.eps_start, self.eps_end, self.eps_steps)
        self.steps_done = 0
        self.last_epsilon = self.eps_start

    def reset_episode(self) -> None:
        self.steps_done = 0

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = True,
        epsilon_override: Optional[float] = None,
    ) -> torch.Tensor:
        state = state.unsqueeze(0).unsqueeze(1)
        sample = random.random()
        if epsilon_override is not None:
            eps_threshold = epsilon_override
        else:
            eps_threshold = self.epsilon_schedule.value(self.steps_done, training=training)
        self.last_epsilon = eps_threshold
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return torch.tensor([self.policy_net(state).argmax()], device=self.device, dtype=torch.long)
        return torch.tensor([random.randrange(self.action_number)], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward) -> None:
        self.memory.push(state, action, next_state, reward)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self) -> Optional[Tuple[float, float]]:
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        nfns = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.batch_size, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.view(self.batch_size, -1)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = mse_q_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item(), state_action_values.detach().mean().item()

    def optimize_double_dqn(self) -> Optional[Tuple[float, float]]:
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        nfns = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.batch_size, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        _, next_state_action = self.policy_net(state_batch).max(1, keepdim=True)

        next_state_values = torch.zeros(self.batch_size, device=self.device).view(self.batch_size, -1)
        out = self.target_net(non_final_next_states)
        next_state_values[non_final_mask] = out.gather(1, next_state_action[non_final_mask])

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = mse_q_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item(), state_action_values.detach().mean().item()
