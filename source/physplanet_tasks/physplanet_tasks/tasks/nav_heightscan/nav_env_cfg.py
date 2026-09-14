# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared GT height-scan navigation environment configuration."""

from __future__ import annotations

import math
import os
import json
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .commands import NavigationCommandCfg
from .mdp.soft_soil import SoftSoilActionCfg


# ── 地形路径解析 ──

def _physplanet_root() -> str:
    """Return the workspace root from this task module."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))


def _resolve_terrain_dir() -> str:
    """从环境变量 ROVER_TERRAIN 解析地形目录绝对路径。

    __file__ = .../physplanet/source/physplanet_tasks/physplanet_tasks/tasks/nav_heightscan/nav_env_cfg.py
    需回溯 6 层 dirname 到 physplanet/，再 join 到 terrain_generation/terrain/<name>
    """
    terrain_name = os.environ.get("ROVER_TERRAIN", "rl_demo/heightscan/01_easy")
    base = os.path.join(
        _physplanet_root(), "source", "physplanet_assets", "terrain_generation", "terrain", terrain_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Terrain dir not found: {base}\n"
            f"Set ROVER_TERRAIN env var to a valid terrain name.")
    return base


# ── height_scan 默认参数（可配置）──

HEIGHT_SCAN_SIZE = (5.0, 5.0)     # 扫描范围（米），yaw 对齐以车为中心
HEIGHT_SCAN_RES = 0.2             # 分辨率（米/cell）→ 26×26 = 676 维
HEIGHT_OFFSET = 0.40
GOAL_RADIUS = 0.8


##
# Scene
##

@configclass
class NavHeightscanSceneCfg(InteractiveSceneCfg):
    """全局单地形 + 多 env 布局。terrain 与 rocks 分两个 prim spawn。"""

    # Terrain and rocks are authored once. V2+ resets each cloned rover inside
    # the same map coordinates; physics replication isolates the rollouts.
    terrain: TerrainImporterCfg = MISSING
    obstacles: AssetBaseCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(intensity=0.0, radius=50.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 80.0)),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    actions: ActionTerm = MISSING
    # 可选的 Bekker-Wong 软土支撑，action_dim=0。置上后地形会切到无碰撞版本，
    # 法向支撑完全由它提供。声明顺序即 apply 顺序，必须排在驱动 term 之后。
    soft_soil: SoftSoilActionCfg | None = None


##
# Observations
##

@configclass
class ObservationsCfg:
    """观测：height_scan + 导航辅助 obs + last_action。"""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(
            func=mdp.distance_to_target_euclidean,
            params={"command_name": "target_pose"}, scale=0.11)
        heading = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={"command_name": "target_pose"}, scale=1.0 / math.pi)
        height_scan = ObsTerm(
            func=mdp.HeightScanGT,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scan_size": HEIGHT_SCAN_SIZE,
                "scan_res": HEIGHT_SCAN_RES,
                "height_offset": HEIGHT_OFFSET,
                "terrain_dir": "",       # __post_init__ 注入
                "terrain_size": 100.0,   # __post_init__ 注入
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Commands
##

@configclass
class CommandsCfg:
    """导航目标 command term。"""
    target_pose: NavigationCommandCfg = NavigationCommandCfg()


##
# Events
##

@configclass
class EventCfg:
    """reset 时随机出生位置和朝向。"""
    reset_state = EventTerm(
        func=mdp.ResetRootStateRover,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_dir": "",
            "terrain_size": 100.0,
            "spawn_jitter": 2.0,
            "spawn_clearance": 1.2,
            "spawn_search_radius": 4.0,
            "spawn_margin": 3.0,
            "z_offset": 0.5,
        },
    )


##
# Navigation rewards
##

@configclass
class RewardsCfg:
    progress_to_target = RewTerm(
        func=mdp.progress_to_target_reward, weight=2.0,
        params={"command_name": "target_pose"})
    reached_target = RewTerm(
        func=mdp.reached_target, weight=5.0,
        params={"command_name": "target_pose", "threshold": GOAL_RADIUS})
    oscillation = RewTerm(
        func=mdp.oscillation_penalty, weight=-0.05, params={})
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty, weight=-1.5,
        params={"command_name": "target_pose"})
    collision = RewTerm(
        func=mdp.collision_penalty, weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    rock_collision = RewTerm(
        func=mdp.RockCollisionPenalty, weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_names": ["base_link", ".*wheel.*"],
            "terrain_dir": "",
            "terrain_size": 100.0,
            "clearance": 0.85,
        })
    rock_proximity = RewTerm(
        func=mdp.RockProximityPenalty, weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_names": ["base_link", ".*wheel.*"],
            "terrain_dir": "",
            "terrain_size": 100.0,
            "clearance": 0.85,
            "warning_distance": 0.8,
        })
    clearance_progress = None
    out_of_bounds = RewTerm(
        func=mdp.out_of_bounds_penalty, weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_size": 100.0,
            "margin": 3.0,
        })
    far_from_target = RewTerm(
        func=mdp.far_from_target_reward, weight=-2.0,
        params={"command_name": "target_pose"})
    reverse_motion = None
    direction_change = None
    idle_while_far = None


##
# Terminations
##

@configclass
class TerminationsCfg:
    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    is_success = DoneTerm(
        func=mdp.is_success,
        params={"command_name": "target_pose", "threshold": GOAL_RADIUS})
    far_from_target = DoneTerm(
        func=mdp.far_from_target,
        params={"command_name": "target_pose"})
    collision = DoneTerm(
        func=mdp.collision_with_obstacles,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    rock_collision = DoneTerm(
        func=mdp.RockCollisionGT,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_names": ["base_link", ".*wheel.*"],
            "terrain_dir": "",
            "terrain_size": 100.0,
            "clearance": 0.85,
        })
    out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_size": 100.0,
            "margin": 3.0,
        })


##
# Environment
##

@configclass
class NavHeightscanEnvCfg(ManagerBasedRLEnvCfg):
    """Shared height-scan navigation task configuration."""

    scene: NavHeightscanSceneCfg = NavHeightscanSceneCfg(num_envs=64, env_spacing=10.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**21,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )

    def __post_init__(self):
        # 解析地形路径
        terrain_dir = _resolve_terrain_dir()

        profile_path = os.path.join(terrain_dir, "terrain_profile.json")
        profile = {}
        if os.path.isfile(profile_path):
            with open(profile_path) as f:
                profile = json.load(f)
        sky = profile.get("sky", "mars")
        if sky == "lunar":
            self.scene.dome_light.spawn = sim_utils.DomeLightCfg(color=(0.0, 0.0, 0.0), intensity=0.0)
            self.scene.sphere_light.spawn = sim_utils.SphereLightCfg(
                intensity=5000.0, radius=50.0, color_temperature=5500.0, enable_color_temperature=True,
            )
        else:
            mars_sky = os.path.join(
                _physplanet_root(), "source", "physplanet_assets", "terrain_generation",
                "models", "textures", "mars", "mars_sky.png",
            )
            self.scene.dome_light.spawn = sim_utils.DomeLightCfg(
                color_temperature=4500.0, intensity=100.0, enable_color_temperature=True,
                texture_file=mars_sky, texture_format="latlong",
            )

        # 读 terrain_size
        size_path = os.path.join(terrain_dir, "terrain_size.txt")
        terrain_size = 100.0
        if os.path.exists(size_path):
            with open(size_path) as f:
                terrain_size = float(f.read().strip())

        # 注入 scene 字段。软土模式下地形不能有碰撞体，否则刚性接触和 Bekker 法向力
        # 双重支撑，车会被顶起来；岩石仍保留碰撞作为兜底。
        terrain_usd = "terrain_only_nocollide.usd" if self.actions.soft_soil else "terrain_only.usd"
        terrain_usd_path = os.path.join(terrain_dir, terrain_usd)
        if not os.path.isfile(terrain_usd_path):
            raise FileNotFoundError(f"Terrain USD not found: {terrain_usd_path}")
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/terrain", terrain_type="usd", usd_path=terrain_usd_path)
        self.scene.obstacles = AssetBaseCfg(
            prim_path="/World/terrain/obstacles",
            spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(terrain_dir, "rocks_merged.usd")))

        # 注入 command 的 terrain_dir
        self.commands.target_pose.terrain_dir = terrain_dir
        self.commands.target_pose.terrain_size = terrain_size
        navigation = profile.get("navigation", {})
        target_range = navigation.get("target_distance_range")
        if target_range is not None:
            if len(target_range) != 2 or target_range[0] <= 0 or target_range[0] > target_range[1]:
                raise ValueError(f"Invalid target_distance_range in {profile_path}: {target_range}")
            self.commands.target_pose.target_distance_range = tuple(map(float, target_range))
        if self.commands.target_pose.use_scenario_profile:
            scenario_weights = navigation.get("scenario_weights")
            if scenario_weights is None:
                scenario_weights = (1.0, 0.0, 0.0)
            if len(scenario_weights) != 3 or sum(scenario_weights) <= 0:
                raise ValueError(f"Invalid scenario_weights in {profile_path}: {scenario_weights}")
            self.commands.target_pose.scenario_weights = tuple(map(float, scenario_weights))

        if self.events.reset_state is not None:
            self.events.reset_state.params["terrain_dir"] = terrain_dir
            self.events.reset_state.params["terrain_size"] = terrain_size
        self.rewards.out_of_bounds.params["terrain_size"] = terrain_size
        self.terminations.out_of_bounds.params["terrain_size"] = terrain_size
        rock_terms = [
            self.rewards.rock_proximity,
            self.rewards.rock_collision,
            self.terminations.rock_collision,
        ]
        if self.rewards.clearance_progress is not None:
            rock_terms.append(self.rewards.clearance_progress)
        for term in rock_terms:
            term.params["terrain_dir"] = terrain_dir
            term.params["terrain_size"] = terrain_size

        # 注入 height_scan obs 的 terrain_dir + terrain_size（该项可能被删除）
        if self.observations.policy.height_scan is not None:
            self.observations.policy.height_scan.params["terrain_dir"] = terrain_dir
            self.observations.policy.height_scan.params["terrain_size"] = terrain_size
        semantic_field = getattr(self.observations.policy, "semantic_scan", None)
        if semantic_field is not None:
            semantic_field.params["terrain_dir"] = terrain_dir
            semantic_field.params["terrain_size"] = terrain_size

        # 仿真/控制参数。控制周期恒为 0.2s；软土模式把物理步细到 0.01s——Bekker 法向力
        # 是显式外力，dt=1/30 下法向刚度会数值发散，0.01s 是 mars_sim 移植时标定过的步长。
        if self.actions.soft_soil is not None:
            self.actions.soft_soil.terrain_dir = terrain_dir
            self.actions.soft_soil.terrain_size = terrain_size
            self.sim.dt = 1.0 / 100.0
            self.decimation = 20
            self.sim.gravity = (0.0, 0.0, -3.72)   # 火星重力，与土参数标定环境一致
        else:
            self.sim.dt = 1.0 / 30.0
            self.decimation = 6
        self.sim.render_interval = self.decimation
        self.episode_length_s = 150
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-6.0, -6.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # sensor 更新周期 = 控制周期
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation

        print(f"[NavHeightscan] terrain_dir={terrain_dir}, terrain_size={terrain_size}, "
              f"num_envs={self.scene.num_envs}")
