"""Zhurong specialization of the shared navigation task."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from rover_lab_assets import ZHURONG_CFG

from ... import mdp
from ...nav_env_cfg import NavHeightscanEnvCfg
from .ackermann import (
    ZhurongAckermannActionCfg,
    ZhurongCommittedSpeedCurvatureActionCfg,
    ZhurongSpeedCurvatureActionCfg,
)

ZHURONG_FOOTPRINT_RADIUS = 1.15


@configclass
class ZhurongNavHeightscanEnvCfg(NavHeightscanEnvCfg):
    def __post_init__(self):
        self.scene.robot = ZHURONG_CFG.replace(prim_path="{ENV_REGEX_NS}/zhurong")
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/zhurong/base_link", history_length=3
        )
        self.actions.actions = ZhurongAckermannActionCfg(asset_name="robot")
        self.observations.policy.height_scan.params["height_offset"] = 0.40
        if self.events.reset_state is not None:
            self.events.reset_state.params["z_offset"] = 0.5
        rock_terms = [
            self.rewards.rock_proximity,
            self.rewards.rock_collision,
            self.terminations.rock_collision,
        ]
        if self.rewards.clearance_progress is not None:
            rock_terms.append(self.rewards.clearance_progress)
        for term in rock_terms:
            term.params["body_names"] = ["base_link"]
            term.params["clearance"] = ZHURONG_FOOTPRINT_RADIUS
        super().__post_init__()


@configclass
class ZhurongNavHeightscanV1EnvCfg(ZhurongNavHeightscanEnvCfg):
    """Speed-curvature navigation baseline with forward and reverse motion."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.actions = ZhurongSpeedCurvatureActionCfg(asset_name="robot")
        self.observations.policy.forward_velocity = ObsTerm(
            func=mdp.base_forward_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=2.5,
        )
        self.observations.policy.yaw_rate = ObsTerm(
            func=mdp.base_yaw_rate,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=1.0,
        )
        self.rewards.oscillation = None
        self.rewards.action_change = RewTerm(func=mdp.action_change_penalty, weight=-0.01)
        self.rewards.time = RewTerm(func=mdp.time_penalty, weight=-0.005)


@configclass
class ZhurongNavHeightscanV2EnvCfg(ZhurongNavHeightscanV1EnvCfg):
    """Scenario-sampled navigation with forward and reverse speed-curvature control."""

    def __post_init__(self):
        self.commands.target_pose.use_scenario_profile = True
        self.commands.target_pose.sample_spawn_and_goal = True
        self.commands.target_pose.max_resample_attempts = 400
        self.commands.target_pose.path_clearance = ZHURONG_FOOTPRINT_RADIUS
        self.commands.target_pose.spawn_clearance = 1.25
        self.commands.target_pose.rock_safety_margin = 1.25
        self.events.reset_state = None
        super().__post_init__()
        self.rewards.action_change = None
        self.rewards.reverse_motion = None
        self.rewards.direction_change = RewTerm(
            func=mdp.direction_change_penalty, weight=-0.01, params={"threshold": 0.25}
        )
        self.rewards.idle_while_far = RewTerm(
            func=mdp.idle_while_far_penalty,
            weight=-0.025,
            params={"command_name": "target_pose", "speed_threshold": 0.1, "distance_threshold": 1.5},
        )
        self.rewards.far_from_target = None
        self.rewards.angle_to_target = None
        self.rewards.collision = None
        self.terminations.collision = None
        self.terminations.far_from_target = None


@configclass
class ZhurongNavHeightscanV3EnvCfg(ZhurongNavHeightscanV2EnvCfg):
    """V3: committed drive direction with a single RockGT clearance-progress reward."""

    def __post_init__(self):
        self.rewards.clearance_progress = RewTerm(
            func=mdp.RockClearanceProgressReward,
            weight=4.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_names": ["base_link", ".*wheel.*"],
                "terrain_dir": "",
                "terrain_size": 100.0,
                "clearance": 0.85,
                "warning_distance": 0.8,
            },
        )
        super().__post_init__()
        self.actions.actions = ZhurongCommittedSpeedCurvatureActionCfg(asset_name="robot")
        self.observations.policy.controller_state = ObsTerm(
            func=mdp.committed_direction_state,
            params={"action_name": "actions"},
        )
        self.rewards.direction_change = None
        self.rewards.idle_while_far = None


@configclass
class ZhurongNavHeightscanV6EnvCfg(ZhurongNavHeightscanV3EnvCfg):
    """V6: feasible recovery cases with costs based on physical rover motion."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.clearance_progress = None
        self.rewards.reverse_motion = RewTerm(
            func=mdp.reverse_motion_penalty,
            weight=-0.015,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_reverse_velocity": self.actions.actions.max_reverse_velocity,
            },
        )
        self.rewards.idle_while_far = RewTerm(
            func=mdp.low_speed_while_far_penalty,
            weight=-0.01,
            params={
                "command_name": "target_pose",
                "asset_cfg": SceneEntityCfg("robot"),
                "speed_threshold": 0.04,
                "distance_threshold": 1.5,
            },
        )
        self.rewards.far_from_target = RewTerm(
            func=mdp.far_from_target_reward,
            weight=-2.0,
            params={"command_name": "target_pose", "margin": 2.5},
        )
        self.terminations.far_from_target = DoneTerm(
            func=mdp.far_from_target,
            params={"command_name": "target_pose", "margin": 2.5},
        )


##
# 软土系列：V7 定义土壤扫描结构 / V10–V12 = Baseline / +Semantic / +Physics 三条件
##

# 土壤视野比 height_scan（5m/0.2m）更大更粗：材料块的尺度大于局部几何采样，
# 因此使用 9×9 网格查询前方的物理参数。每个点提供 n0 和 phi 两个通道。
SOIL_SCAN_SIZE = (8.0, 8.0)
SOIL_SCAN_RES = 1.0
# n0 和 phi（弧度）均已处于适合策略输入的数值范围。
SOIL_CHANNEL_SCALE = (1.0, 1.0)
# 安全线位于 gravel（静态沉陷 ~10mm）与 sand（~17mm）之间。
SAFE_SINKAGE = 0.015
# terrain_class_map 是 1-based 的，按土层从硬到软排列。
# 取 3 = 最软一类（softsoil_zw 设计里的 P5 GRC-3，静态沉陷 28mm，会陷车）
# 排除出生/目标，但路径内部仍会穿过——"避开软土"由此成为有代价的真实决策。
MAX_SOIL_CLASS = 3
# 陷车判据按当前 loose-soil 参数的静态沉陷范围保留，最终是否触发由动态响应决定。
BOGGED_SINKAGE_RATIO = 0.13

# The three learning conditions use the same observation width.  The two extra
# fields are zero-filled in the baseline and in the semantic-only condition.
SEMANTIC_SCAN_SIZE = (8.0, 8.0)
SEMANTIC_SCAN_RES = 1.0
SEMANTIC_COUNT = 4


@configclass
class ZhurongNavHeightscanV7EnvCfg(ZhurongNavHeightscanV6EnvCfg):
    """V7: V6 + Bekker-Wong 软土。观测里只加本体感受的逐轮下陷，不给任何前视土壤信息。"""

    def __post_init__(self):
        self.actions.soft_soil = mdp.SoftSoilActionCfg(asset_name="robot")
        self.commands.target_pose.max_soil_class = MAX_SOIL_CLASS
        super().__post_init__()
        self.observations.policy.wheel_sinkage = ObsTerm(
            func=mdp.wheel_sinkage, params={"action_name": "soft_soil"}
        )
        # 权重量级参考：一次成功回合的 progress+reached 合计约 +6.5。
        # 越过安全线的 sand(~17mm)/loose(~34mm) 长时间碾过会累积明显负值。
        self.rewards.sinkage = RewTerm(
            func=mdp.sinkage_penalty, weight=-1.0,
            params={"action_name": "soft_soil", "safe_sinkage": SAFE_SINKAGE},
        )
        self.rewards.wheel_slip = RewTerm(
            func=mdp.wheel_slip_penalty, weight=-0.5,
            params={"action_name": "soft_soil", "slip_threshold": 0.4},
        )
        self.terminations.bogged_down = DoneTerm(
            func=mdp.BoggedDown,
            params={
                "action_name": "soft_soil",
                "asset_cfg": SceneEntityCfg("robot"),
                "sinkage_ratio": BOGGED_SINKAGE_RATIO,
                "slip_threshold": 0.6,
                "speed_threshold": 0.03,
                "patience_steps": 15,
            },
        )


def _add_physics_aware_fields(cfg, semantic_enabled: bool, physics_enabled: bool) -> None:
    """Add dimension-matched semantic and physical local fields."""
    cfg.observations.policy.semantic_scan = ObsTerm(
        func=mdp.SemanticScanGT,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scan_size": SEMANTIC_SCAN_SIZE,
            "scan_res": SEMANTIC_SCAN_RES,
            "terrain_dir": "",
            "terrain_size": 100.0,
            "semantic_count": SEMANTIC_COUNT,
            "enabled": semantic_enabled,
        },
    )
    cfg.observations.policy.soil_scan = ObsTerm(
        func=mdp.SoilScanGT,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_name": "soft_soil",
            "scan_size": SOIL_SCAN_SIZE,
            "scan_res": SOIL_SCAN_RES,
            "channel_scale": SOIL_CHANNEL_SCALE,
            "decorrelate_shift": None,
            "enabled": physics_enabled,
        },
        clip=(-3.0, 3.0),
    )


@configclass
class ZhurongNavHeightscanV10EnvCfg(ZhurongNavHeightscanV7EnvCfg):
    """HeightScan baseline with fixed-width zero semantic and physics fields."""

    def __post_init__(self):
        _add_physics_aware_fields(self, semantic_enabled=False, physics_enabled=False)
        super().__post_init__()


@configclass
class ZhurongNavHeightscanV11EnvCfg(ZhurongNavHeightscanV10EnvCfg):
    """HeightScan plus the aligned one-hot semantic field."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.semantic_scan.params["enabled"] = True


@configclass
class ZhurongNavHeightscanV12EnvCfg(ZhurongNavHeightscanV10EnvCfg):
    """HeightScan plus semantic and aligned physical fields."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.semantic_scan.params["enabled"] = True
        self.observations.policy.soil_scan.params["enabled"] = True




