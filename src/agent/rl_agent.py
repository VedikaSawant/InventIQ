# agent/rl_agent.py

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import Categorical


# =========================================================
# SHARED CONSTANTS
# =========================================================

ACTION_LEVELS = 11
ORDER_STEP = 10
MAX_ORDER = 100

HORIZON = 7

STOCK_IDX = 0
FORECAST_IDXS = slice(1, 8)


# =========================================================
# ACTOR-CRITIC NETWORK
# =========================================================

class ActorCritic(nn.Module):

    def __init__(self, state_dim, n_actions, hidden=128):

        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden, n_actions)

        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):

        features = self.backbone(x)

        logits = self.actor(features)

        value = self.critic(features).squeeze(-1)

        return logits, value

    def get_action(self, obs):

        logits, value = self(obs)

        dist = Categorical(logits=logits)

        action = dist.sample()

        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):

        logits, value = self(obs)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        return log_probs, value, entropy


# =========================================================
# PPO AGENT
# =========================================================

class PPOAgent:

    def __init__(
        self,
        state_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        n_epochs=4,
        batch_size=64,
        rollout_len=512,
    ):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_len = rollout_len

        self.policy = ActorCritic(
            state_dim,
            n_actions
        )

        self.optimizer = Adam(
            self.policy.parameters(),
            lr=lr
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, env, total_timesteps=200000):

        obs, _ = env.reset()

        timestep = 0

        print("Starting PPO training...")

        while timestep < total_timesteps:

            rollout = self._collect_rollout(env, obs)

            obs = rollout["next_obs"]

            self._ppo_update(rollout)

            timestep += self.rollout_len

            # 🔥 ADD THIS LOG
            if timestep % 2048 == 0:
                print(f"PPO timestep: {timestep}/{total_timesteps}")

        print("PPO training finished.")

    # =====================================================
    # ROLLOUT
    # =====================================================

    def _collect_rollout(self, env, obs):

        buf = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

        for _ in range(self.rollout_len):

            obs_t = torch.from_numpy(obs).float().unsqueeze(0)

            action, log_prob, value = self.policy.get_action(obs_t)

            next_obs, reward, terminated, truncated, _ = env.step(
                action.item()
            )

            done = terminated or truncated

            buf["obs"].append(obs)
            buf["actions"].append(action.item())
            buf["log_probs"].append(log_prob.item())
            buf["rewards"].append(reward)
            buf["dones"].append(done)
            buf["values"].append(value.item())

            obs = next_obs

            if done:
                obs, _ = env.reset()

        # bootstrap
        with torch.no_grad():

            _, last_value = self.policy(
                torch.from_numpy(obs).float().unsqueeze(0)
            )

        advantages = self._compute_gae(
            buf["rewards"],
            buf["values"],
            buf["dones"],
            last_value.item()
        )

        buf["advantages"] = advantages

        buf["returns"] = (
            np.array(advantages)
            + np.array(buf["values"])
        ).tolist()

        buf["next_obs"] = obs

        return buf

    # =====================================================
    # GAE
    # =====================================================

    def _compute_gae(
        self,
        rewards,
        values,
        dones,
        last_value,
    ):

        advantages = np.zeros(len(rewards))

        gae = 0

        for t in reversed(range(len(rewards))):

            next_val = (
                last_value
                if t == len(rewards) - 1
                else values[t + 1]
            )

            delta = (
                rewards[t]
                + self.gamma * next_val * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages[t] = gae

        return advantages.tolist()

    def save(self, filename):

        import os

        os.makedirs("outputs/models", exist_ok=True)

        torch.save(
            self.policy.state_dict(),
            f"outputs/models/{filename}"
        )

        print(f"PPO model saved to outputs/models/{filename}")

    def load(self, filename):

        path = f"outputs/models/{filename}"

        self.policy.load_state_dict(
            torch.load(path, map_location="cpu")
        )

        print(f"PPO model loaded from {path}")

    # =====================================================
    # PPO UPDATE
    # =====================================================

    def _ppo_update(self, rollout):

        obs = torch.tensor(
            np.array(rollout["obs"]),
            dtype=torch.float32
        )

        actions = torch.tensor(
            rollout["actions"],
            dtype=torch.long
        )

        old_lp = torch.tensor(
            rollout["log_probs"],
            dtype=torch.float32
        )

        advantages = torch.tensor(
            rollout["advantages"],
            dtype=torch.float32
        )

        returns = torch.tensor(
            rollout["returns"],
            dtype=torch.float32
        )

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):

            indices = np.random.permutation(len(obs))

            for start in range(
                0,
                len(obs),
                self.batch_size
            ):

                idx = indices[start:start+self.batch_size]

                new_lp, values, entropy = self.policy.evaluate(
                    obs[idx],
                    actions[idx]
                )

                ratio = (new_lp - old_lp[idx]).exp()

                surr1 = ratio * advantages[idx]

                surr2 = ratio.clamp(
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * advantages[idx]

                policy_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                value_loss = nn.functional.mse_loss(
                    values,
                    returns[idx]
                )

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    0.5
                )

                self.optimizer.step()

    # =====================================================
    # INFERENCE
    # =====================================================

    def predict(self, obs):

        self.policy.eval()

        with torch.no_grad():

            obs_t = torch.from_numpy(obs).float().unsqueeze(0)

            logits, _ = self.policy(obs_t)

            return logits.argmax(dim=-1).item()


# =========================================================
# BASELINE POLICIES
# =========================================================

def units_to_action(units):

    units = float(np.clip(units, 0, MAX_ORDER))

    action = int(round(units / ORDER_STEP))

    return int(np.clip(action, 0, ACTION_LEVELS - 1))


class ReorderPolicy:

    def __init__(
        self,
        reorder_point=20,
        max_level=80,
        max_stock=500
    ):

        self.s = reorder_point
        self.S = max_level
        self.max_stock = max_stock

    def act(self, obs):

        stock = obs[STOCK_IDX] * self.max_stock

        if stock <= self.s:

            return units_to_action(
                self.S - stock
            )

        return 0