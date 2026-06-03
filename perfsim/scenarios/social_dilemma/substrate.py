"""Public-goods game as a torchrl EnvBase (the MARL substrate).

N agents act simultaneously in one coupled game. We use the IPPO convention:
`batch_size = [N]` so torchrl treats the agent axis as the batch, a single
parameter-shared policy maps each agent's observation to its contribution, and
the reward COUPLING across agents (everyone shares the multiplied pool) is
computed inside `_step`. This is independent PPO with parameter sharing -- the
standard cheap MARL baseline -- not torchrl's formal multi-group API.

One game step:
  - each agent i contributes a_i in [0, 1] of its (unit) endowment
  - the pool is multiplied by r and split evenly: pool_share = r * mean(a)
  - base payoff: payoff_i = (1 - a_i) + pool_share        (defect dominates)
  - intrinsic type motivation: + kappa * w_i * a_i, so a "civic" agent (high
    w_i) gets private utility from contributing. This is what makes the
    EXOGENOUS type actually drive behavior: at gamma = 0 the equilibrium is
    heterogeneous and tied to type (civic cooperate, selfish defect), giving a
    real type signal for the platform to preserve or crowd out. Without it the
    agents are identical payoff-maximizers and type is inert.
  - the platform mediates via reward shaping with strength gamma:
        reward_i = payoff_i + kappa * w_i * a_i - gamma * (a_i - theta_i)^2
    so gamma = 0 is the public-goods dilemma with type-driven heterogeneity, and
    gamma > 0 pulls each agent toward the platform's prediction theta(x_i).

Observation per agent = [x_i (fixed type/features), prev_mean_contribution],
so the policy has a dynamic signal (the previous round's cooperation level) and
the game is a proper -- if shallow -- MDP rather than a pure contextual bandit.

`theta_pred` (the frozen platform prediction, one scalar per agent) is set by
the owner (`PublicGoodsMARL`) at the top of each epoch via `set_context`; the
substrate itself never queries the platform model.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase


class PublicGoodsSubstrate(EnvBase):
    """Coupled N-agent public-goods game with platform reward-shaping.

    batch_size = [n_agents]; obs (n_agents, D+1), action (n_agents, 1),
    reward (n_agents, 1). The episode runs `horizon` steps then signals done.
    """

    def __init__(
        self,
        x: Tensor,
        theta_pred: Tensor,
        intrinsic_weight: Tensor,
        *,
        r_multiplier: float = 1.6,
        kappa: float = 2.0,
        gamma: float = 0.0,
        horizon: int = 16,
        init_mean: float = 0.5,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        n_agents, d = x.shape
        super().__init__(device=device, batch_size=torch.Size([n_agents]))
        self._dtype = dtype
        self._n = int(n_agents)
        self._d = int(d)
        self.r_multiplier = float(r_multiplier)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.init_mean = float(init_mean)

        self.register_buffer("_x", x.to(device=device, dtype=dtype))
        self.register_buffer("_theta", theta_pred.reshape(-1, 1).to(device=device, dtype=dtype))
        self.register_buffer("_w", intrinsic_weight.reshape(-1, 1).to(device=device, dtype=dtype))
        self._step_count = 0

        obs_dim = self._d + 1
        self.observation_spec = Composite(
            observation=Unbounded(shape=torch.Size([self._n, obs_dim]), dtype=dtype),
            shape=torch.Size([self._n]),
        )
        self.action_spec = Bounded(
            low=0.0, high=1.0, shape=torch.Size([self._n, 1]), dtype=dtype
        )
        self.reward_spec = Composite(
            reward=Unbounded(shape=torch.Size([self._n, 1]), dtype=dtype),
            shape=torch.Size([self._n]),
        )
        self.done_spec = Composite(
            done=Bounded(low=0, high=1, shape=torch.Size([self._n, 1]), dtype=torch.bool),
            terminated=Bounded(low=0, high=1, shape=torch.Size([self._n, 1]), dtype=torch.bool),
            shape=torch.Size([self._n]),
        )

    def set_context(self, theta_pred: Tensor, *, gamma: float | None = None) -> None:
        """Refresh the frozen platform prediction (and optionally gamma) per epoch."""
        self._theta = theta_pred.reshape(-1, 1).to(device=self.device, dtype=self._dtype)
        if gamma is not None:
            self.gamma = float(gamma)

    def _obs(self, mean_contrib: float) -> Tensor:
        col = torch.full((self._n, 1), float(mean_contrib), device=self.device, dtype=self._dtype)
        return torch.cat([self._x, col], dim=1)

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        self._step_count = 0
        obs = self._obs(self.init_mean)
        done = torch.zeros((self._n, 1), dtype=torch.bool, device=self.device)
        return TensorDict(
            {"observation": obs, "done": done, "terminated": done.clone()},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        a = tensordict["action"].reshape(self._n, 1).clamp(0.0, 1.0)
        pool_share = self.r_multiplier * a.mean()
        payoff = (1.0 - a) + pool_share
        intrinsic = self.kappa * self._w * a
        reward = payoff + intrinsic - self.gamma * (a - self._theta) ** 2

        self._step_count += 1
        done = torch.full(
            (self._n, 1), self._step_count >= self.horizon, dtype=torch.bool, device=self.device
        )
        obs = self._obs(float(a.mean()))
        return TensorDict(
            {
                "observation": obs,
                "reward": reward.to(self._dtype),
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        self._rng = torch.manual_seed(0 if seed is None else int(seed))
