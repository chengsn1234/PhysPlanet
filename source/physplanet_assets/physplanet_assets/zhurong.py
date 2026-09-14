####################
## Zhurong Asset
## 功能：祝融号 USD / actuator / articulation 配置
####################

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from . import ZHURONG_ASSETS_DIR


####################
## Actuator 配置
####################

ZHURONG_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            "suspension_steer_F_L_joint", "suspension_steer_F_R_joint",
            "suspension_steer_M_L_joint", "suspension_steer_M_R_joint",
            "suspension_steer_B_L_joint", "suspension_steer_B_R_joint",
        ],
        velocity_limit=10.0,
        effort_limit=10000.0,
        stiffness=1000.0,
        damping=100.0,
    ),
    "throttle_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            "front_wheel_L_joint", "front_wheel_R_joint",
            "middle_wheel_L_joint", "middle_wheel_R_joint",
            "back_wheel_L_joint", "back_wheel_R_joint",
        ],
        effort_limit=10000.0,
        velocity_limit=100.0,
        stiffness=0.0,
        damping=100.0,
    ),
    "suspension_arm_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            "suspension_arm_F_L_joint", "suspension_arm_F_R_joint",
            "suspension_arm_B_L_joint", "suspension_arm_B_R_joint",
        ],
        effort_limit=10000.0,
        velocity_limit=10.0,
        stiffness=5000.0,
        damping=10.0,
    ),
    "suspension_arm_B2_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            "suspension_arm_B2_L_joint", "suspension_arm_B2_R_joint",
        ],
        effort_limit=10000.0,
        velocity_limit=10.0,
        stiffness=0.0,
        damping=10.0,
    ),
}


####################
## Articulation 配置
####################

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # 6轮
        'front_wheel_L_joint': 0.0, 'front_wheel_R_joint': 0.0,
        'middle_wheel_L_joint': 0.0, 'middle_wheel_R_joint': 0.0,
        'back_wheel_L_joint': 0.0, 'back_wheel_R_joint': 0.0,
        # 6转向
        'suspension_steer_F_L_joint': 0.0, 'suspension_steer_F_R_joint': 0.0,
        'suspension_steer_M_L_joint': 0.0, 'suspension_steer_M_R_joint': 0.0,
        'suspension_steer_B_L_joint': 0.0, 'suspension_steer_B_R_joint': 0.0,
        # 4主摇臂（主动悬挂）
        'suspension_arm_F_L_joint': 0.0, 'suspension_arm_F_R_joint': 0.0,
        'suspension_arm_B_L_joint': 0.0, 'suspension_arm_B_R_joint': 0.0,
        # 2副摇臂（被动）
        'suspension_arm_B2_L_joint': 0.0, 'suspension_arm_B2_R_joint': 0.0,
        # PTZ 云台（mars_sim 中是 fixed，这里设为 0）
        # 'PTZYaw_joint': 0.0,
        # 'PTZPitch_joint': 0.0,
    },
)

ZHURONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ZHURONG_ASSETS_DIR}/zhurong.usd",
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
    actuators=ZHURONG_ACTUATOR_CFG,
)
