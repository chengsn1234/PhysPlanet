"""skrl agent construction for continuous-control RoverLab tasks."""

from __future__ import annotations

import copy
import math
from numbers import Number

import gymnasium as gym
import torch
from skrl.agents.torch.base import ExperimentCfg
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.noises.torch import GaussianNoise
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper

from physplanet_rl.models import DeterministicPolicy, GaussianPolicy, QNetwork, ValueNetwork


ON_POLICY_ALGORITHMS = frozenset(("PPO",))
OFF_POLICY_ALGORITHMS = frozenset(("TD3", "SAC"))
SUPPORTED_ALGORITHMS = ON_POLICY_ALGORITHMS | OFF_POLICY_ALGORITHMS

_AGENTS = {
    "PPO": (PPO, PPO_CFG),
    "TD3": (TD3, TD3_CFG),
    "SAC": (SAC, SAC_CFG),
}


class RoverIsaacLabWrapper(IsaacLabWrapper):
    """Preserve Isaac Lab's episodic scalar logs for skrl's TensorBoard writer."""

    def _tensorize_log_scalars(self, infos):
        log_data = infos.get("log") if isinstance(infos, dict) else None
        if not isinstance(log_data, dict):
            return infos
        for key, value in log_data.items():
            if isinstance(value, Number) and not isinstance(value, bool):
                log_data[key] = torch.as_tensor(value, device=self.device)
        return infos

    def reset(self):
        observations, infos = super().reset()
        return observations, self._tensorize_log_scalars(infos)

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = super().step(actions)
        return observations, rewards, terminated, truncated, self._tensorize_log_scalars(infos)


class SampleDeviceRandomMemory(RandomMemory):
    """Keep replay on one device and train from batches on another."""

    def __init__(self, *args, sample_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_device = torch.device(sample_device or self.device)

    def sample_by_index(self, names, *, indexes, mini_batches=1):
        batches = super().sample_by_index(names, indexes=indexes, mini_batches=mini_batches)
        if self._sample_device == self.device:
            return batches
        return [[item.to(self._sample_device, non_blocking=True) if item is not None else None for item in batch]
                for batch in batches]


def _spaces(env):
    observation_space = env.observation_space
    action_space = env.action_space
    if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
        raise ValueError("RoverLab skrl agents require a flat Box policy observation space")
    if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
        raise ValueError("RoverLab skrl agents require a flat Box action space")
    # Isaac Lab reports unbounded action spaces even when action terms use normalized commands.
    normalized_actions = gym.spaces.Box(low=-1.0, high=1.0, shape=action_space.shape, dtype=action_space.dtype)
    return observation_space, normalized_actions


def _network(cfg, name):
    try:
        network = cfg["models"][name]["network"][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Missing models.{name}.network[0] in agent config") from exc
    return network["layers"], network.get("activations", "leaky_relu")


def _build_models(algorithm, cfg, observation_space, action_space, device):
    policy_layers, policy_activation = _network(cfg, "policy")
    if algorithm in ON_POLICY_ALGORITHMS:
        value_layers, value_activation = _network(cfg, "value")
        policy_cfg = cfg["models"]["policy"]
        return {
            "policy": GaussianPolicy(
                observation_space, action_space, device,
                mlp_layers=policy_layers,
                mlp_activation=policy_activation,
                initial_log_std=policy_cfg.get("initial_log_std", -0.5),
                min_log_std=policy_cfg.get("min_log_std", -5.0),
                max_log_std=policy_cfg.get("max_log_std", 0.0),
            ),
            "value": ValueNetwork(
                observation_space, action_space, device,
                mlp_layers=value_layers, mlp_activation=value_activation,
            ),
        }

    critic_layers, critic_activation = _network(cfg, "critic")
    if algorithm == "TD3":
        policy = DeterministicPolicy(
            observation_space, action_space, device,
            mlp_layers=policy_layers, mlp_activation=policy_activation,
        )
        critic_1 = QNetwork(
            observation_space, action_space, device,
            mlp_layers=critic_layers, mlp_activation=critic_activation,
        )
        critic_2 = QNetwork(
            observation_space, action_space, device,
            mlp_layers=critic_layers, mlp_activation=critic_activation,
        )
        return {
            "policy": policy,
            "target_policy": DeterministicPolicy(
                observation_space, action_space, device,
                mlp_layers=policy_layers, mlp_activation=policy_activation,
            ),
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
            "target_critic_2": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
        }

    if algorithm == "SAC":
        policy_cfg = cfg["models"]["policy"]
        return {
            "policy": GaussianPolicy(
                observation_space, action_space, device,
                mlp_layers=policy_layers,
                mlp_activation=policy_activation,
                initial_log_std=policy_cfg.get("initial_log_std", -0.5),
                min_log_std=policy_cfg.get("min_log_std", -5.0),
                max_log_std=policy_cfg.get("max_log_std", 0.0),
            ),
            "critic_1": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
            "critic_2": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
            "target_critic_1": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
            "target_critic_2": QNetwork(
                observation_space, action_space, device,
                mlp_layers=critic_layers, mlp_activation=critic_activation,
            ),
        }
    raise ValueError(f"Unsupported algorithm {algorithm}")


def _agent_cfg(cfg, algorithm, disable_logging, device):
    values = copy.deepcopy(cfg["agent"])
    values.pop("class", None)
    experiment = values.pop("experiment")
    if disable_logging:
        experiment["write_interval"] = 0
        experiment["checkpoint_interval"] = 0
    values["experiment"] = ExperimentCfg(**experiment)

    if algorithm == "TD3":
        noise = values.pop("exploration_noise", None)
        if noise:
            values["exploration_noise"] = GaussianNoise
            values["exploration_noise_kwargs"] = {**noise, "device": device}
        smoothing = values.pop("smooth_regularization_noise", None)
        if smoothing:
            values["smooth_regularization_noise"] = GaussianNoise
            values["smooth_regularization_noise_kwargs"] = {**smoothing, "device": device}

    cfg_type = _AGENTS[algorithm][1]
    unknown = set(values) - set(cfg_type.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unsupported skrl {algorithm} settings: {sorted(unknown)}")
    return values


def _build_memory(env, cfg, algorithm, agent_cfg):
    if algorithm in ON_POLICY_ALGORITHMS:
        memory_size = cfg.get("memory", {}).get("memory_size", agent_cfg["rollouts"])
        memory_size = agent_cfg["rollouts"] if memory_size in (-1, None) else int(memory_size)
        return RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    memory_cfg = cfg.get("memory", {})
    total_size = int(memory_cfg.get("total_size", 200000))
    memory_size = max(1, math.ceil(total_size / env.num_envs))
    storage_device = memory_cfg.get("device", "cpu")
    print(
        f"[INFO] replay: total={total_size}, per_env={memory_size}, "
        f"storage={storage_device}, sample={env.device}"
    )
    return SampleDeviceRandomMemory(
        memory_size=memory_size,
        num_envs=env.num_envs,
        device=storage_device,
        sample_device=env.device,
    )


def algorithm_name(cfg):
    algorithm = cfg["agent"].get("class", "PPO").upper()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algorithm}. Available: {sorted(SUPPORTED_ALGORITHMS)}")
    return algorithm


def build_agent(env, cfg: dict, disable_logging: bool = False):
    """Build a supported skrl agent from a task-owned YAML config."""
    algorithm = algorithm_name(cfg)
    observation_space, action_space = _spaces(env)
    models = _build_models(algorithm, cfg, observation_space, action_space, env.device)
    agent_cfg = _agent_cfg(cfg, algorithm, disable_logging, env.device)
    memory = _build_memory(env, cfg, algorithm, agent_cfg)
    agent_type = _AGENTS[algorithm][0]
    print(f"[INFO] algorithm={algorithm}, obs_dim={observation_space.shape[0]}, action_dim={action_space.shape[0]}")
    return agent_type(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )


def eval_actions(agent, observations):
    """Return deterministic evaluation actions without exploration noise."""
    preprocessor = getattr(agent, "_observation_preprocessor", None)
    if preprocessor is not None:
        observations = preprocessor(observations)
    actions, outputs = agent.policy.act({"observations": observations}, role="policy")
    return outputs.get("mean_actions", actions)
