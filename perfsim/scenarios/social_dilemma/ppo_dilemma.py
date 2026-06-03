"""PublicGoodsMARL: the perfsim env whose inner loop is a torchrl PPO run.

This is the RL version of the social-dilemma scenario. The population is a
parameter-shared PPO policy (IPPO) acting in `PublicGoodsSubstrate`. One perfsim
epoch = freeze the platform prediction theta(x), train the policy for
`n_steps` collect-and-update cycles in the substrate, then read off each agent's
converged contribution y_i in [0, 1]. The platform then trains on (x_i, y_i) in
the outer Simulator loop, and its new prediction re-enters next epoch through the
substrate's reward shaping (strength gamma). gamma = 0 is the un-mediated
public-goods dilemma; gamma > 0 makes the platform a behavioral mediator.

The policy PERSISTS across epochs by default (the population keeps learning as
theta shifts), which matches "train the agents, then introduce the predictor."
Set `reset_policy_each_epoch=True` to retrain from scratch each epoch instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from perfsim.core.environment import AgentBased
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.scenarios.social_dilemma.substrate import PublicGoodsSubstrate


class PublicGoodsMARL(AgentBased):
    """N-agent public-goods PP env with a PPO population (torchrl)."""

    max_meaningful_epoch_size = float("inf")

    def __init__(
        self,
        x: Tensor,
        type_baseline: Tensor,
        *,
        gamma: float = 0.0,
        r_multiplier: float = 1.6,
        kappa: float = 2.0,
        horizon: int = 16,
        hidden: int = 64,
        policy_lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
        gae_gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reset_policy_each_epoch: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (N, D); got {tuple(x.shape)}")
        self._x = x.to(device=device, dtype=dtype)
        self._type_baseline = type_baseline.reshape(-1).to(device=device, dtype=dtype)
        self._n, self._d = self._x.shape
        self.gamma = float(gamma)
        self.r_multiplier = float(r_multiplier)
        self.kappa = float(kappa)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.policy_lr = float(policy_lr)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coeff = float(entropy_coeff)
        self.ppo_epochs = int(ppo_epochs)
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset_policy_each_epoch = bool(reset_policy_each_epoch)
        self._device = torch.device(device)
        self._dtype = dtype
        self._agent_idx = torch.arange(self._n)

        # theta_0 is unknown at construction; start the substrate with a neutral
        # prediction (0.5). The owner refreshes it every epoch in run().
        self._theta_pred = torch.full((self._n,), 0.5, dtype=dtype, device=self._device)
        self.substrate = PublicGoodsSubstrate(
            self._x,
            self._theta_pred,
            self._type_baseline,
            r_multiplier=self.r_multiplier,
            kappa=self.kappa,
            gamma=self.gamma,
            horizon=self.horizon,
            device=self._device,
            dtype=self._dtype,
        )
        self._build_policy(seed=0)

    # ---- properties --------------------------------------------------------

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_agents(self) -> int:
        return self._n

    # ---- policy / optimizer ------------------------------------------------

    def _build_policy(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        obs_dim = self._d + 1
        actor_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 2),  # loc, scale (pre-softplus)
            NormalParamExtractor(),
        ).to(self._device, self._dtype)
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": 0.0, "high": 1.0},
            return_log_prob=True,
        ).to(self._device)

        value_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
        ).to(self._device, self._dtype)
        self.critic = ValueOperator(value_net, in_keys=["observation"]).to(self._device)

        self.adv_module = GAE(
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=self.entropy_coeff,
        )
        self._opt = torch.optim.Adam(self.loss_module.parameters(), lr=self.policy_lr)

    # ---- Environment API ---------------------------------------------------

    def reset(self, seed: int = 0) -> None:
        self.substrate.set_seed(int(seed))
        if self.reset_policy_each_epoch or not hasattr(self, "actor"):
            self._build_policy(seed=seed)

    def _train_policy(self, n_cycles: int) -> None:
        for _ in range(n_cycles):
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                td = self.substrate.rollout(self.horizon, self.actor)
            for _ in range(self.ppo_epochs):
                self.adv_module(td)
                loss_td = self.loss_module(td)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()

    @torch.no_grad()
    def _readout(self) -> Tensor:
        """Each agent's converged contribution: deterministic action, time-averaged."""
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.substrate.rollout(self.horizon, self.actor)
        return td["action"].reshape(self._n, self.horizon).mean(dim=1).clamp(0.0, 1.0)

    def _epoch(self, model: Model, n_steps: int) -> Data:
        with torch.no_grad():
            theta_pred = model(self._x).reshape(-1).to(self._dtype)
        self._theta_pred = theta_pred
        self.substrate.set_context(theta_pred, gamma=self.gamma)
        if self.reset_policy_each_epoch:
            self._build_policy(seed=0)
        self._train_policy(n_steps)
        y = self._readout()
        return {
            "x": self._x.clone(),
            "y": y.reshape(-1, 1).clone(),
            "agent_idx": self._agent_idx.clone(),
        }

    def run(self, model: Model, n_steps: int) -> Data:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive int; got {n_steps!r}")
        return self._epoch(model, n_steps)

    def step(self, model: Model) -> Data:
        return self._epoch(model, 1)

    def sample(self, model: Model) -> Data:
        """Peek: refresh context and read off behavior WITHOUT training the policy."""
        with torch.no_grad():
            theta_pred = model(self._x).reshape(-1).to(self._dtype)
        self.substrate.set_context(theta_pred, gamma=self.gamma)
        return {
            "x": self._x.clone(),
            "y": self._readout().reshape(-1, 1).clone(),
            "agent_idx": self._agent_idx.clone(),
        }
