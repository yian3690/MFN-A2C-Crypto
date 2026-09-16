"""Dirichlet actor policy for long-only portfolio weights on a simplex."""

from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim


class DirichletDistribution(DiagGaussianDistribution):
    """SB3-compatible Dirichlet distribution over unit-sum actions.

    SB3 dispatches continuous distributions by concrete distribution type.
    This class keeps the continuous-distribution marker inherited from
    ``DiagGaussianDistribution``, but replaces every probabilistic operation
    with a Dirichlet distribution. The second network return value is an
    empty compatibility parameter; no Gaussian standard deviation is used.
    """

    def __init__(self, action_dim: int, min_concentration: float = 1e-4):
        super().__init__(action_dim)
        self.min_concentration = float(min_concentration)
        self.concentration: th.Tensor | None = None

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0.0,
    ) -> tuple[nn.Module, nn.Parameter]:
        del log_std_init
        concentration_net = nn.Linear(latent_dim, self.action_dim)
        compatibility_parameter = nn.Parameter(
            th.empty(0),
            requires_grad=False,
        )
        return concentration_net, compatibility_parameter

    def proba_distribution(
        self,
        raw_concentration: th.Tensor,
        unused_parameter: th.Tensor | None = None,
    ) -> "DirichletDistribution":
        del unused_parameter
        self.concentration = (
            F.softplus(raw_concentration) + self.min_concentration
        )
        self.distribution = th.distributions.Dirichlet(self.concentration)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Protect saved/rounded actions from log(0), then restore the simplex.
        safe_actions = actions.clamp_min(th.finfo(actions.dtype).tiny)
        safe_actions = safe_actions / safe_actions.sum(dim=-1, keepdim=True)
        return self.distribution.log_prob(safe_actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        # The true mode lies on a boundary when any concentration <= 1.
        # The mean is always valid and is a stable deterministic allocation.
        assert self.concentration is not None
        return self.concentration / self.concentration.sum(
            dim=-1,
            keepdim=True,
        )

    def actions_from_params(
        self,
        raw_concentration: th.Tensor,
        unused_parameter: th.Tensor | None = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        self.proba_distribution(raw_concentration, unused_parameter)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        raw_concentration: th.Tensor,
        unused_parameter: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(
            raw_concentration,
            unused_parameter,
        )
        return actions, self.log_prob(actions)


class SimplexActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy whose actions are portfolio weights directly."""

    def _build(self, lr_schedule) -> None:
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("SimplexActorCriticPolicy requires a Box action space.")

        action_dim = get_action_dim(self.action_space)
        self.action_dist = DirichletDistribution(action_dim)
        super()._build(lr_schedule)

        # ActorCriticPolicy only recognizes built-in distribution classes.
        # DirichletDistribution inherits the Gaussian marker solely so SB3
        # creates action_net for us. It does not use a Gaussian log_std;
        # remove the empty compatibility attribute to avoid logging std=NaN.
        del self.log_std

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
    ) -> DirichletDistribution:
        """Create the Dirichlet distribution from actor latent features."""
        raw_concentration = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_concentration)
