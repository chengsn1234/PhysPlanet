import gymnasium as gym

from ... import agents
from .env_cfg import HunterNavHeightscanEnvCfg

_SKRL_CFGS = {
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    "skrl_td3_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
}

gym.register(
    id="Isaac-HunterNavHeightscan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HunterNavHeightscanEnvCfg, **_SKRL_CFGS},
)
