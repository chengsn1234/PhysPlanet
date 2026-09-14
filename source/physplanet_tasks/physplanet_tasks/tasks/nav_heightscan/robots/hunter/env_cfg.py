"""Hunter specialization of the shared navigation task."""

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from physplanet_assets import HUNTER_CFG

from ...nav_env_cfg import NavHeightscanEnvCfg
from .ackermann import HunterAckermannActionCfg


@configclass
class HunterNavHeightscanEnvCfg(NavHeightscanEnvCfg):
    def __post_init__(self):
        self.scene.robot = HUNTER_CFG.replace(prim_path="{ENV_REGEX_NS}/hunter")
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/hunter/base_link", history_length=3
        )
        self.actions.actions = HunterAckermannActionCfg(asset_name="robot")
        self.observations.policy.height_scan.params["height_offset"] = 0.40
        self.events.reset_state.params["z_offset"] = 0.45
        for term in (self.rewards.rock_proximity, self.rewards.rock_collision, self.terminations.rock_collision):
            term.params["body_names"] = ["base_link", "fr_left_link", "fr_right_link", "re_left_link", "re_right_link"]
        super().__post_init__()
