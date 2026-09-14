import gymnasium as gym

from ... import agents
from .env_cfg import (
    ZhurongNavHeightscanEnvCfg,
    ZhurongNavHeightscanV1EnvCfg,
    ZhurongNavHeightscanV2EnvCfg,
    ZhurongNavHeightscanV3EnvCfg,
    ZhurongNavHeightscanV6EnvCfg,
    ZhurongNavHeightscanV7EnvCfg,
    ZhurongNavHeightscanV10EnvCfg,
    ZhurongNavHeightscanV11EnvCfg,
    ZhurongNavHeightscanV12EnvCfg,
)

_SOFT_SOIL_ENTRY = "physplanet_tasks.tasks.nav_heightscan.soft_soil_env:SoftSoilNavEnv"

_SKRL_CFGS = {
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    "skrl_td3_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
}

gym.register(
    id="Isaac-ZhurongNavHeightscan-v10",
    entry_point=_SOFT_SOIL_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": ZhurongNavHeightscanV10EnvCfg, **_SKRL_CFGS},
)

gym.register(
    id="Isaac-ZhurongNavHeightscan-v11",
    entry_point=_SOFT_SOIL_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": ZhurongNavHeightscanV11EnvCfg, **_SKRL_CFGS},
)

gym.register(
    id="Isaac-ZhurongNavHeightscan-v12",
    entry_point=_SOFT_SOIL_ENTRY,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": ZhurongNavHeightscanV12EnvCfg, **_SKRL_CFGS},
)
