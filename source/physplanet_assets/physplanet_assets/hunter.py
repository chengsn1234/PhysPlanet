####################
## Hunter Asset
## 功能：Hunter USD / actuator / articulation 配置
####################

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from . import HUNTER_ASSETS_DIR


####################
## Actuator 配置
####################

HUNTER_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["fr_steer_left_joint", "fr_steer_right_joint"],
        velocity_limit=2.0,
        effort_limit=50.0,
        stiffness=500.0,
        damping=50.0,
    ),
    "throttle_joints": ImplicitActuatorCfg(
        joint_names_expr=["re_left_joint", "re_right_joint", "fr_left_joint", "fr_right_joint"],
        effort_limit=10000.0,
        velocity_limit=15.0,
        stiffness=0.0,
        damping=20.0,
    ),
}

HUNTER_SUS_ACTUATOR_CFG = {
    **HUNTER_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_prismatic_joint"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=20000.0,
        damping=200.0,
    ),
    "roll_joint": ImplicitActuatorCfg(
        joint_names_expr=["fr_roll_joint"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=100.0,
        damping=10.0,
    ),
}


####################
## Articulation 配置
####################

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        're_left_joint': 0.0,
        're_right_joint': 0.0,
        'fr_steer_left_joint': 0.0,
        'fr_steer_right_joint': 0.0,
        'fr_right_joint': 0.0,
        'fr_left_joint': 0.0,
        'fr_roll_joint': 0.0,
    },
)

HUNTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HUNTER_ASSETS_DIR}/hunter.usd",
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.04,
            rest_offset=0.01,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=1.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=HUNTER_ACTUATOR_CFG,
)

HUNTER_SUS_CFG = HUNTER_CFG.replace(
    spawn=HUNTER_CFG.spawn.replace(
        usd_path=f"{HUNTER_ASSETS_DIR}/hunter.usd",
        activate_contact_sensors=True,
    ),
    init_state=_ZERO_INIT_STATES.replace(
        joint_pos={
            **_ZERO_INIT_STATES.joint_pos,
            're_left_prismatic_joint': 0.0,
            're_right_prismatic_joint': 0.0,
        }
    ),
    actuators=HUNTER_SUS_ACTUATOR_CFG,
)
