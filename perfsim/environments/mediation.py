"""AI-mediated human data channel as a performative map D(theta).

A fixed raw human population (x0, y) is observed by the platform only AFTER it
passes through an AI assistant psi (the Mediator). The assistant rewrites the
"style" columns z while preserving the "fundamental" columns c; it may also be
conditioned on the platform's current prediction theta(x) ("rewrite to raise the
score"). The platform then trains on the mediated data, so what it sees next
round depends on what it deployed this round: a performative loop.

This mirrors StrategicLinearWorld (Perdomo): a fixed population, recomputed each
round under the current theta. The difference is the response map. Strategic =
rational best-response (epsilon * w); here the response is a mediation operator
psi applied to the style columns. Recursion/accumulation across rounds is the
job of the retrain driver (perfsim.scenarios.ai_mediated), NOT this env: the env
is one-shot D(theta) over the fixed raw population.

The z/c split is a column mask, exactly analogous to strat_features. Labels are
either preserved (y = f(c) attached to the original person) or regenerated from
the mediated artifact (y' = f(z'), e.g. toxicity of the rewritten text).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Iterable, Optional

import torch
from torch import Tensor

from perfsim.core.environment import StatefulDynamics
from perfsim.core.model import Model
from perfsim.core.types import SUPERVISED_SCHEMA, Data, DataSchema
from perfsim.maps._common import validate_strat_features


class Mediator(ABC):
    """The AI assistant psi: a channel that rewrites the style columns of x.

    `score` is the platform's current prediction theta(x) (one value per row),
    passed only when the mediation is platform-conditioned; otherwise None.
    Implementations must leave non-style columns untouched.
    """

    @abstractmethod
    def mediate(
        self,
        x: Tensor,
        *,
        style_mask: Tensor,
        score: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return x_obs: x with the style columns transformed by the channel."""


class IdentityMediator(Mediator):
    """No-op channel: x_obs = x. The null baseline (no mediation)."""

    def mediate(
        self,
        x: Tensor,
        *,
        style_mask: Tensor,
        score: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        return x.clone()


class ContractionMediator(Mediator):
    """Quasi-model assistant: pull the style columns toward a shared target.

    x_z' = (1 - strength) * x_z + strength * target  (+ optional jitter)

    target_mode:
      "centroid": target = population mean of the style columns (generic
        mediation, no platform context). One shared assistant -> everyone drifts
        to the same house style. Saturating / idempotent.
      "winners": target = mean style of the top `top_frac` rows by `score`
        (platform-conditioned: "rewrite to look like what scores well"). The
        target depends on theta via score, so it moves every round -> the
        "predicting from predictions" feedback loop.

    strength is the contraction coefficient lambda in [0, 1]; strength=0 is a
    no-op, strength=1 collapses every style vector onto the target. `jitter`
    adds optional Gaussian noise (std) to the style columns for realism.
    """

    def __init__(
        self,
        strength: float = 0.5,
        *,
        target_mode: str = "centroid",
        top_frac: float = 0.25,
        jitter: float = 0.0,
    ) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength (lambda) must be in [0, 1]; got {strength}")
        if target_mode not in ("centroid", "winners"):
            raise ValueError(
                f"target_mode must be 'centroid' or 'winners'; got {target_mode!r}"
            )
        if not 0.0 < top_frac <= 1.0:
            raise ValueError(f"top_frac must be in (0, 1]; got {top_frac}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0; got {jitter}")
        self.strength = float(strength)
        self.target_mode = target_mode
        self.top_frac = float(top_frac)
        self.jitter = float(jitter)

    def mediate(
        self,
        x: Tensor,
        *,
        style_mask: Tensor,
        score: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        x_obs = x.clone()
        style = x[:, style_mask]
        if self.target_mode == "centroid":
            target = style.mean(dim=0, keepdim=True)
        else:
            if score is None:
                raise ValueError(
                    "ContractionMediator(target_mode='winners') requires a "
                    "platform score; build MediationWorld with "
                    "platform_conditioned=True."
                )
            n = style.shape[0]
            k = max(1, int(round(self.top_frac * n)))
            top_idx = torch.topk(score.reshape(-1), k=k, largest=True).indices
            target = style[top_idx].mean(dim=0, keepdim=True)
        new_style = (1.0 - self.strength) * style + self.strength * target
        if self.jitter > 0.0:
            noise = torch.randn(
                new_style.shape, generator=generator, dtype=new_style.dtype
            )
            new_style = new_style + self.jitter * noise
        x_obs[:, style_mask] = new_style
        return x_obs


class LinearSurrogateMediator(Mediator):
    """Data-calibrated channel: a linear map fit from real (original, rewrite) pairs.

    x_z' = x_z @ weight.T + bias, where (weight, bias) are learned to reproduce a
    real LLM rewrite (e.g. ridge regression of rewrite-embedding on original-
    embedding). Applied once it is mild and affine (preserves separability), but
    applied RECURSIVELY in a self_consuming loop it iterates x_{t+1}=M x_t + b,
    which converges to the map's fixed point at a rate set by the singular values
    of weight -- i.e. recursion is what turns a mild per-pass shift into collapse.
    This is the surrogate that lets the loop run many rounds with no API.
    """

    def __init__(self, weight: Tensor, bias: Tensor) -> None:
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"weight must be square (Dz, Dz); got {tuple(weight.shape)}")
        if bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"bias dim {bias.shape[0]} != weight dim {weight.shape[0]}"
            )
        self.weight = weight
        self.bias = bias

    def mediate(
        self,
        x: Tensor,
        *,
        style_mask: Tensor,
        score: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        x_obs = x.clone()
        style = x[:, style_mask].to(self.weight.dtype)
        x_obs[:, style_mask] = (style @ self.weight.T + self.bias).to(x.dtype)
        return x_obs


LabelFn = Callable[[Tensor], Tensor]


class MediationWorld(StatefulDynamics):
    """D(theta): observe a fixed raw population through an AI mediator psi.

    The population (x0, y) is fixed at init. Each round the style columns are
    re-mediated under the CURRENT theta (via the mediator, optionally
    platform-conditioned), so the observed data tracks theta. Labels are either
    preserved (attached to the original person) or regenerated from the mediated
    features by `label_fn`.
    """

    max_meaningful_epoch_size: ClassVar[int] = 1

    def __init__(
        self,
        x0: Tensor,
        y: Tensor,
        *,
        mediator: Mediator,
        style_features: Iterable[int] | None = None,
        label_mode: str = "preserve",
        label_fn: LabelFn | None = None,
        platform_conditioned: bool = False,
        self_consuming: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if x0.ndim != 2:
            raise ValueError(f"x0 must be 2-D (N, D); got {tuple(x0.shape)}")
        if y.shape[0] != x0.shape[0]:
            raise ValueError(
                f"y leading dim {y.shape[0]} != x0 leading dim {x0.shape[0]}"
            )
        if label_mode not in ("preserve", "regenerate"):
            raise ValueError(
                f"label_mode must be 'preserve' or 'regenerate'; got {label_mode!r}"
            )
        if label_mode == "regenerate" and label_fn is None:
            raise ValueError("label_mode='regenerate' requires label_fn")
        self._x0_init = x0.to(dtype=dtype).clone()  # original population, for reset
        self._x0 = self._x0_init.clone()             # current population (may evolve)
        self._y = y.clone()
        if self._y.ndim == 1:
            self._y = self._y.unsqueeze(-1)
        self._dtype = dtype
        self._n, self._d = self._x0.shape
        self.mediator = mediator
        self.label_mode = label_mode
        self.label_fn = label_fn
        self.platform_conditioned = bool(platform_conditioned)
        # self_consuming: the mediated output replaces the population each step
        # (the recursive model-collapse loop). Otherwise the fixed raw population
        # is re-mediated from scratch each round (the conservative baseline).
        self.self_consuming = bool(self_consuming)
        # Style columns z; the rest are fundamentals c. None => all columns z.
        strat = validate_strat_features(style_features, dim=self._d)
        mask = torch.zeros(self._d, dtype=torch.bool)
        if strat is None:
            mask[:] = True
        else:
            mask[strat] = True
        self._style_mask = mask
        self._agent_idx = torch.arange(self._n)
        self._gen: torch.Generator | None = None

    @property
    def produces_schema(self) -> DataSchema:
        return SUPERVISED_SCHEMA

    @property
    def n_agents(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._d

    @property
    def style_mask(self) -> Tensor:
        return self._style_mask.clone()

    @property
    def raw_data(self) -> Data:
        """The clean, unmediated population (the pre-AI human anchor)."""
        return {
            "x": self._x0_init.clone(),
            "y": self._y.clone(),
            "agent_idx": self._agent_idx.clone(),
        }

    def reset(self, seed: int = 0) -> None:
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))
        self._x0 = self._x0_init.clone()  # restore the original population

    def _score(self, model: Model) -> Tensor | None:
        if not self.platform_conditioned:
            return None
        with torch.no_grad():
            out = model(self._x0)
        return out.detach().reshape(-1).to(self._dtype)

    def _mediate(self, model: Model) -> Data:
        if self._gen is None:
            self.reset(seed=0)
        score = self._score(model)
        x_obs = self.mediator.mediate(
            self._x0,
            style_mask=self._style_mask,
            score=score,
            generator=self._gen,
        ).to(self._dtype)
        if self.label_mode == "preserve":
            y_obs = self._y.clone()
        else:
            assert self.label_fn is not None
            y_obs = self.label_fn(x_obs)
            if y_obs.ndim == 1:
                y_obs = y_obs.unsqueeze(-1)
        return {"x": x_obs, "y": y_obs, "agent_idx": self._agent_idx.clone()}

    def sample(self, model: Model) -> Data:
        """Peek at D(theta): mediate without mutating the population."""
        return self._mediate(model)

    def step(self, model: Model) -> Data:
        """Advance one step; under self_consuming the output replaces the population."""
        data = self._mediate(model)
        if self.self_consuming:
            self._x0 = data["x"].detach().clone()
            if self.label_mode == "regenerate":
                self._y = data["y"].detach().clone()
        return data
