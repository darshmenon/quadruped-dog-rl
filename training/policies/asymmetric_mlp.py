"""Asymmetric actor-critic policy for Stable-Baselines3 PPO.

Actor: ``obs['policy']`` only (deployable proprioception).
Critic: ``obs['critic']`` = policy ∥ privileged lin_vel (training-only).

Inspired by DreamWaQ / FR-Net asymmetric critic without CENet.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import get_device


class AsymmetricMlpExtractor(nn.Module):
    """Separate policy/value MLP stacks with different input feature dims."""

    def __init__(
        self,
        pi_feature_dim: int,
        vf_feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [])
            vf_layers = net_arch.get("vf", [])
        else:
            pi_layers = vf_layers = net_arch

        self.policy_net = self._build_stack(
            pi_feature_dim, pi_layers, activation_fn).to(device)
        self.value_net = self._build_stack(
            vf_feature_dim, vf_layers, activation_fn).to(device)
        self.latent_dim_pi = pi_feature_dim if not pi_layers else pi_layers[-1]
        self.latent_dim_vf = vf_feature_dim if not vf_layers else vf_layers[-1]

    @staticmethod
    def _build_stack(
        in_dim: int, layers: list[int], activation_fn: type[nn.Module]
    ) -> nn.Sequential:
        modules: list[nn.Module] = []
        last = in_dim
        for dim in layers:
            modules.extend([nn.Linear(last, dim), activation_fn()])
            last = dim
        return nn.Sequential(*modules)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with separate actor/critic observation routing."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        policy_obs_key: str = "policy",
        critic_obs_key: str = "critic",
        *args,
        **kwargs,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "AsymmetricActorCriticPolicy requires a Dict observation space "
                f"with '{policy_obs_key}' and '{critic_obs_key}' keys")
        self.policy_obs_key = policy_obs_key
        self.critic_obs_key = critic_obs_key
        policy_space = observation_space.spaces[policy_obs_key]
        critic_space = observation_space.spaces[critic_obs_key]
        self.actor_observation_space = policy_space
        self.critic_observation_space = critic_space

        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.vf_features_extractor = FlattenExtractor(critic_space)
        self._build(lr_schedule)

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        return self.features_extractor_class(
            self.actor_observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            pi_feature_dim=self.features_dim,
            vf_feature_dim=self.vf_features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _obs_tensor(self, obs: PyTorchObs, key: str) -> PyTorchObs:
        if isinstance(obs, dict):
            return obs[key]
        if key == self.policy_obs_key:
            return obs
        raise ValueError(
            f"Expected Dict observation with key '{key}', got shape "
            f"{getattr(obs, 'shape', type(obs))}")

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> th.Tensor:
        fe = features_extractor or self.features_extractor
        policy_obs = self._obs_tensor(obs, self.policy_obs_key)
        preprocessed = preprocess_obs(
            policy_obs,
            self.actor_observation_space,
            normalize_images=self.normalize_images,
        )
        return fe(preprocessed)

    def extract_features_critic(self, obs: PyTorchObs) -> th.Tensor:
        critic_obs = self._obs_tensor(obs, self.critic_obs_key)
        preprocessed = preprocess_obs(
            critic_obs,
            self.critic_observation_space,
            normalize_images=self.normalize_images,
        )
        return self.vf_features_extractor(preprocessed)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def forward(
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        pi_features = self.extract_features(obs, self.features_extractor)
        vf_features = self.extract_features_critic(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        pi_features = self.extract_features(obs, self.features_extractor)
        vf_features = self.extract_features_critic(obs)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        vf_features = self.extract_features_critic(obs)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf)
