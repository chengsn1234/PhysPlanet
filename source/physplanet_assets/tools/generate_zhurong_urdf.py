#!/usr/bin/env python3
# 祝融号 URDF 生成脚本
####################
## 功能：
## 1. 用数据驱动方式生成祝融号 6 轮火星车 URDF
## 2. 轮组拓扑: 前轮(chassis→arm→steer→wheel), 中后轮(chassis→arm_B→arm_B2→steer→wheel)
## 3. 左右镜像: y 值取反, ixy/iyz 取反
####################

import copy
import os
from dataclasses import dataclass


# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZHURONG_ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "physplanet_assets", "zhurong_assets")
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
ZHURONG_DESCRIPTION_URDF_DIR = os.path.join(REPO_ROOT, "ros2_ws", "src", "zhurong_description", "urdf")
MESH_DIR = "package://zhurong_description/meshes"

OUTPUT_URDF = os.path.join(ZHURONG_ASSETS_DIR, "zhurong.urdf")
SYNCED_OUTPUT_URDF = os.path.join(ZHURONG_DESCRIPTION_URDF_DIR, "zhurong.urdf")

# 物理参数
CHASSIS_MASS = 115.0
WHEEL_RADIUS = 0.15
WHEEL_WIDTH = 0.2
WHEEL_MASS = 9.87

# 材质颜色 (RGB, URDF 里 alpha=1)
MATERIAL_COLORS = {
    "chassis_mat": (0.51, 0.38, 0.15),
    "arm_mat": (0.79, 0.82, 0.93),
    "steer_mat": (0.79, 0.82, 0.93),
    "wheel_mat": (0.53, 0.51, 0.49),
    "mast_mat": (0.51, 0.38, 0.15),
    "ptz_mat": (0.79, 0.82, 0.93),
    "cam_mat": (1.0, 0.74, 0.59),
}

# mesh 到材质映射 (bind_zhurong_materials.py 也用这份，保持一致)
MESH_TO_MATERIAL = {
    "chassis": "chassis_mat", "base_link": "chassis_mat",
    "wheel_LF": "wheel_mat", "wheel_LM": "wheel_mat", "wheel_LB": "wheel_mat",
    "wheel_RF": "wheel_mat", "wheel_RM": "wheel_mat", "wheel_RB": "wheel_mat",
    "front_wheel": "wheel_mat", "middle_wheel": "wheel_mat", "back_wheel": "wheel_mat",
    "suspension_arm": "arm_mat", "suspension_steer": "steer_mat", "steerSupport": "steer_mat",
    "leftMainRocker": "arm_mat", "rightMainRocker": "arm_mat",
    "leftSubRocker": "arm_mat", "rightSubRocker": "arm_mat",
    "PTZSupport": "mast_mat", "PTZBase": "ptz_mat", "PTZYaw": "ptz_mat", "PTZPitch": "ptz_mat",
    "ZEDNav": "cam_mat", "LIDAR": "ptz_mat", "obsZED": "cam_mat",
}


## ---- dataclass 定义 ----

@dataclass
class Link:
    name: str
    comment: str = ""  # XML 注释（输出时放在 link 标签上方）
    inertial_origin: str = "0 0 0"
    mass: float = 0.0
    ixx: float = 0.0; ixy: float = 0.0; ixz: float = 0.0
    iyy: float = 0.0; iyz: float = 0.0; izz: float = 0.0
    visual_origin: str = "0 0 0"
    visual_rpy: str = "0 0 0"
    mesh: str = ""
    material: str = ""
    collision: str = ""  # 格式: "cylinder R L" 或 "box W H D" 或 "mesh path"
    collision_origin: str = "0 0 0"
    collision_rpy: str = "0 0 0"


@dataclass
class Joint:
    name: str
    jtype: str  # fixed, revolute, continuous
    parent: str
    child: str
    origin: str = "0 0 0"
    origin_rpy: str = "0 0 0"
    axis: str = ""
    lower: float = None
    upper: float = None
    effort: float = 0.0
    velocity: float = 0.0
    damping: float = 0.0
    friction: float = 0.0


## ---- XML 格式化 ----

def _rgba(mat_name):
    r, g, b = MATERIAL_COLORS[mat_name]
    return f"{r} {g} {b} 1"


def _fv(v):
    # 格式化浮点数：消除 -0.0，避免不必要科学计数法
    if v == 0:
        return "0"
    if isinstance(v, int):
        return str(v)
    # 极小值（相机惯性等）保留科学计数法
    if abs(v) < 1e-6:
        return f"{v}"
    return f"{v:.10f}".rstrip('0').rstrip('.')


def _fmt_link(link):
    # 生成单个 link 的 XML
    lines = []

    ## 注释（空行 + <!-- -->）
    if link.comment:
        lines.append('')
        lines.append(f'  <!-- {link.comment} -->')

    ## 空 link（无 inertial/visual/collision）→ 自闭合
    if not (link.mass > 0 or link.mesh or link.collision):
        lines.append(f'  <link name="{link.name}"/>')
        return "\n".join(lines)

    lines.append(f'  <link name="{link.name}">')

    ## inertial
    if link.mass > 0:
        lines.append(f'    <inertial>')
        lines.append(f'      <origin xyz="{link.inertial_origin}" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{link.mass}"/>')
        lines.append(f'      <inertia ixx="{_fv(link.ixx)}" ixy="{_fv(link.ixy)}" ixz="{_fv(link.ixz)}" '
                     f'iyy="{_fv(link.iyy)}" iyz="{_fv(link.iyz)}" izz="{_fv(link.izz)}"/>')
        lines.append(f'    </inertial>')

    ## visual
    if link.mesh:
        lines.append(f'    <visual>')
        lines.append(f'      <origin xyz="{link.visual_origin}" rpy="{link.visual_rpy}"/>')
        lines.append(f'      <geometry>')
        lines.append(f'        <mesh filename="{MESH_DIR}/{link.mesh}"/>')
        lines.append(f'      </geometry>')
        lines.append(f'      <material name="{link.material}"><color rgba="{_rgba(link.material)}"/></material>')
        lines.append(f'    </visual>')

    ## collision
    if link.collision.startswith("cylinder"):
        _, r, l = link.collision.split()
        lines.append(f'    <collision>')
        lines.append(f'      <origin xyz="{link.collision_origin}" rpy="{link.collision_rpy}"/>')
        lines.append(f'      <geometry>')
        lines.append(f'        <cylinder radius="{r}" length="{l}"/>')
        lines.append(f'      </geometry>')
        lines.append(f'    </collision>')
    elif link.collision.startswith("box"):
        _, s = link.collision.split(None, 1)
        lines.append(f'    <collision>')
        lines.append(f'      <origin xyz="{link.collision_origin}" rpy="{link.collision_rpy}"/>')
        lines.append(f'      <geometry>')
        lines.append(f'        <box size="{s}"/>')
        lines.append(f'      </geometry>')
        lines.append(f'    </collision>')
    elif link.collision.startswith("mesh"):
        _, path = link.collision.split(None, 1)
        lines.append(f'    <collision>')
        lines.append(f'      <origin xyz="{link.collision_origin}" rpy="{link.collision_rpy}"/>')
        lines.append(f'      <geometry>')
        lines.append(f'        <mesh filename="{MESH_DIR}/{path}"/>')
        lines.append(f'      </geometry>')
        lines.append(f'    </collision>')

    lines.append(f'  </link>')
    return "\n".join(lines)


def _fmt_joint(joint):
    # 生成单个 joint 的 XML
    lines = [f'  <joint name="{joint.name}" type="{joint.jtype}">']
    lines.append(f'    <parent link="{joint.parent}"/>')
    lines.append(f'    <child link="{joint.child}"/>')
    if joint.origin_rpy != "0 0 0":
        lines.append(f'    <origin xyz="{joint.origin}" rpy="{joint.origin_rpy}"/>')
    else:
        lines.append(f'    <origin xyz="{joint.origin}" rpy="0 0 0"/>')

    if joint.jtype in ("revolute", "continuous"):
        limit_parts = []
        if joint.lower is not None:
            limit_parts.append(f'lower="{joint.lower}"')
        if joint.upper is not None:
            limit_parts.append(f'upper="{joint.upper}"')
        limit_parts.append(f'effort="{joint.effort}"')
        limit_parts.append(f'velocity="{joint.velocity}"')
        lines.append(f'    <limit {" ".join(limit_parts)}/>')
    if joint.axis:
        lines.append(f'    <axis xyz="{joint.axis}"/>')
    if joint.jtype in ("revolute", "continuous") and (joint.damping > 0 or joint.friction > 0):
        lines.append(f'    <dynamics damping="{joint.damping}" friction="{joint.friction}"/>')

    lines.append(f'  </joint>')
    return "\n".join(lines)


## ---- 组件生成 ----

def _add_chassis(links, joints):
    # base_link + chassis
    links.append(Link(name="base_link", comment="Base link"))
    joints.append(Joint(name="base_c_joint", jtype="fixed", parent="base_link", child="chassis"))
    links.append(Link(
        name="chassis", comment="Chassis",
        inertial_origin="-0.055 0 0.048", mass=CHASSIS_MASS,
        ixx=9.0675, ixy=0.00018, ixz=-0.0011, iyy=11.333, iyz=0.00003, izz=15.759,
        mesh="base_link.STL", material="chassis_mat",
        collision="box 0.25 1.6 0.502", collision_origin="0.85 0 0.105",
    ))


# 轮组参数表（按位置定义，左右镜像由 _add_wheel_assemblies 处理）
# 每个位置定义: arm, steer(s), wheel(s)
# 前轮拓扑: chassis → arm_F → steer_F → front_wheel
# 中后轮拓扑: chassis → arm_B → arm_B2(sub_rocker) → steer_M → middle_wheel
#                                                          → steer_B → back_wheel

_WHEEL_SITES = {
    "arm": {"lower": -0.4, "upper": 0.4, "effort": 10000, "velocity": 10, "axis": "0 1 0"},
    "sub": {"lower": -1.2, "upper": 0.1, "effort": 10000, "velocity": 10, "axis": "0 1 0"},
    "steer": {"lower": -0.6, "upper": 0.6, "effort": 10000, "velocity": 10, "axis": "0 0 1"},
    "wheel": {"effort": 10000, "velocity": 100, "axis": "0 1 0"},
}


def _add_wheel_assemblies(links, joints):
    # 生成 6 个轮组（3 位置 x 2 侧）
    # 定义每个位置的摇臂参数（左侧值，右侧由镜像函数翻转 y/ixy/iyz）
    positions = {
        "F": {
            "arm": {
                "link": Link(
                    name="suspension_arm_F",  # _L/_R 后缀在镜像时添加
                    inertial_origin="0.374 0.093 0", mass=5.916,
                    ixx=0.045, ixy=-0.127, ixz=0, iyy=0.528, iyz=0, izz=0.567,
                    mesh="leftMainRockerF_Link.STL", material="arm_mat",
                ),
                "joint_origin": "0 0.425 0",
                "parent": "chassis",
            },
            "steer": {
                "link": Link(
                    name="suspension_steer_F",
                    inertial_origin="0 0.074 -0.163", mass=3.466,
                    ixx=0.03, ixy=0, ixz=0, iyy=0.023, iyz=0.003, izz=0.01,
                    mesh="steerSupport_LF_Link.STL", material="steer_mat",
                ),
                "joint_origin": "0.775 0.227 -0.04",
            },
            "wheel": {
                "link": Link(
                    name="front_wheel",
                    mass=WHEEL_MASS,
                    ixx=0.101, ixy=0, ixz=0, iyy=0.127, iyz=0, izz=0.101,
                    mesh="wheel_LF_Link.STL", material="wheel_mat",
                    collision=f"cylinder {WHEEL_RADIUS} {WHEEL_WIDTH}",
                    collision_rpy="1.5708 0 0",
                ),
                "joint_origin": "0 0 -0.21",
            },
        },
        "M": {
            # 中轮挂在 sub_rocker 上，这里定义 sub_rocker 的参数
            "arm": {
                "link": Link(
                    name="suspension_arm_B",  # 后摇臂（中后轮共享）
                    inertial_origin="-0.268 0.078 -0.035", mass=4.492,
                    ixx=0.014, ixy=0.022, ixz=-0.013, iyy=0.11, iyz=0.003, izz=0.111,
                    mesh="leftMainRockerB_Link.STL", material="arm_mat",
                ),
                "joint_origin": "0 0.4755 0",
                "parent": "chassis",
            },
            "sub_rocker": {
                "link": Link(
                    name="suspension_arm_B2",
                    inertial_origin="0 0.04 0.028", mass=6.22,
                    ixx=0.014, ixy=0, ixz=0, iyy=0.449, iyz=0.002, izz=0.446,
                    mesh="leftSubRocker_Link.STL", material="arm_mat",
                ),
                "joint_origin": "-0.3875 0.147 -0.05",
                "parent": "suspension_arm_B",  # 挂在 arm_B 下
            },
            "steer": {
                "link": Link(
                    name="suspension_steer_M",
                    inertial_origin="0 0.074 -0.163", mass=3.466,
                    ixx=0.03, ixy=0, ixz=0, iyy=0.023, iyz=0.003, izz=0.01,
                    mesh="steerSupport_LM_Link.STL", material="steer_mat",
                ),
                "joint_origin": "0.3875 0.03 0.01",
                "parent": "suspension_arm_B2",  # 挂在 sub_rocker 上
            },
            "wheel": {
                "link": Link(
                    name="middle_wheel",
                    mass=WHEEL_MASS,
                    ixx=0.101, ixy=0, ixz=0, iyy=0.127, iyz=0, izz=0.101,
                    mesh="wheel_LM_Link.STL", material="wheel_mat",
                    collision=f"cylinder {WHEEL_RADIUS} {WHEEL_WIDTH}",
                    collision_rpy="1.5708 0 0",
                ),
                "joint_origin": "0 0 -0.21",
            },
        },
        "B": {
            # 后轮也挂在 sub_rocker 上（与 M 共享 arm_B + arm_B2）
            # arm 和 sub_rocker 由 M 位置定义，这里只定义 steer 和 wheel
            "steer": {
                "link": Link(
                    name="suspension_steer_B",
                    inertial_origin="0 0.074 -0.163", mass=3.466,
                    ixx=0.03, ixy=0, ixz=0, iyy=0.023, iyz=0.003, izz=0.01,
                    mesh="steerSupport_LB_Link.STL", material="steer_mat",
                ),
                "joint_origin": "-0.3875 0.03 0.01",
                "parent": "suspension_arm_B2",
            },
            "wheel": {
                "link": Link(
                    name="back_wheel",
                    mass=WHEEL_MASS,
                    ixx=0.101, ixy=0, ixz=0, iyy=0.127, iyz=0, izz=0.101,
                    mesh="wheel_LB_Link.STL", material="wheel_mat",
                    collision=f"cylinder {WHEEL_RADIUS} {WHEEL_WIDTH}",
                    collision_rpy="1.5708 0 0",
                ),
                "joint_origin": "0 0 -0.21",
            },
        },
    }

    sides = {
        "L": {"suffix": "_L", "sign": 1, "wheel_rpy": "0 0 0", "label": "LEFT"},
        "R": {"suffix": "_R", "sign": -1, "wheel_rpy": "3.14159 0 0", "label": "RIGHT"},
    }

    # 注释名映射
    arm_comments = {"F": "FRONT ROCKER", "M": "BACK ROCKER"}
    steer_comments = {"F": "FRONT STEER", "M": "MIDDLE STEER", "B": "BACK STEER"}
    wheel_comments = {"F": "FRONT WHEEL", "M": "MIDDLE WHEEL", "B": "BACK WHEEL"}

    for side_key, side in sides.items():
        sign = side["sign"]
        suffix = side["suffix"]
        side_name = side["label"]

        for pos_key, pos_data in positions.items():
            ## arm（仅 F 和 M 位置定义了 arm）
            if "arm" in pos_data:
                arm = pos_data["arm"]
                arm_link = _mirror_link(arm["link"], side)
                arm_link.comment = f"{side_name} {arm_comments[pos_key]}"
                arm_parent = arm["parent"] + suffix if arm["parent"] != "chassis" else "chassis"
                links.append(arm_link)
                joints.append(Joint(
                    name=f"{arm_link.name}_joint", jtype="revolute",
                    parent=arm_parent, child=arm_link.name,
                    origin=_flip_y(arm["joint_origin"], sign),
                    axis=_WHEEL_SITES["arm"]["axis"],
                    lower=_WHEEL_SITES["arm"]["lower"], upper=_WHEEL_SITES["arm"]["upper"],
                    effort=_WHEEL_SITES["arm"]["effort"], velocity=_WHEEL_SITES["arm"]["velocity"],
                ))

            ## sub_rocker（仅 M 位置）
            if "sub_rocker" in pos_data:
                sub = pos_data["sub_rocker"]
                sub_link = _mirror_link(sub["link"], side)
                sub_link.comment = f"{side_name} SUB ROCKER"
                sub_parent = sub["parent"] + suffix
                links.append(sub_link)
                joints.append(Joint(
                    name=f"{sub_link.name}_joint", jtype="revolute",
                    parent=sub_parent, child=sub_link.name,
                    origin=_flip_y(sub["joint_origin"], sign),
                    axis=_WHEEL_SITES["sub"]["axis"],
                    lower=_WHEEL_SITES["sub"]["lower"], upper=_WHEEL_SITES["sub"]["upper"],
                    effort=_WHEEL_SITES["sub"]["effort"], velocity=_WHEEL_SITES["sub"]["velocity"],
                ))

            ## steer
            if "steer" in pos_data:
                steer = pos_data["steer"]
                steer_link = _mirror_link(steer["link"], side)
                steer_link.comment = f"{side_name} {steer_comments[pos_key]}"
                steer_parent = (steer.get("parent", "") + suffix) if steer.get("parent") else _get_steer_parent(pos_key, suffix)
                links.append(steer_link)
                joints.append(Joint(
                    name=f"{steer_link.name}_joint", jtype="revolute",
                    parent=steer_parent, child=steer_link.name,
                    origin=_flip_y(steer["joint_origin"], sign),
                    axis=_WHEEL_SITES["steer"]["axis"],
                    lower=_WHEEL_SITES["steer"]["lower"], upper=_WHEEL_SITES["steer"]["upper"],
                    effort=_WHEEL_SITES["steer"]["effort"], velocity=_WHEEL_SITES["steer"]["velocity"],
                ))

            ## wheel
            if "wheel" in pos_data:
                wh = pos_data["wheel"]
                wh_link = _mirror_link(wh["link"], side)
                wh_link.visual_rpy = side["wheel_rpy"]
                wh_link.comment = f"{side_name} {wheel_comments[pos_key]}"
                steer_name = f"suspension_steer_{pos_key}{suffix}"
                links.append(wh_link)
                joints.append(Joint(
                    name=f"{wh_link.name}_joint", jtype="continuous",
                    parent=steer_name, child=wh_link.name,
                    origin=_flip_y(wh["joint_origin"], sign),
                    axis=_WHEEL_SITES["wheel"]["axis"],
                    effort=_WHEEL_SITES["wheel"]["effort"], velocity=_WHEEL_SITES["wheel"]["velocity"],
                ))


def _get_steer_parent(pos_key, suffix):
    # steer 挂在谁下面？
    if pos_key == "F":
        return f"suspension_arm_F{suffix}"
    else:
        return f"suspension_arm_B2{suffix}"


def _flip_y(origin_str, sign):
    # 翻转 origin 中 y 值（sign=-1 时取反），消除 -0.0
    parts = origin_str.split()
    y = float(parts[1]) * sign
    if y == int(y):
        y = int(y)
    parts[1] = str(y)
    return " ".join(parts)


def _mirror_link(link, side):
    # 镜像一个 link：添加侧后缀、翻转 y/ixy/iyz、替换 mesh 名前缀
    new = copy.deepcopy(link)
    s = side["suffix"]
    sign = side["sign"]

    new.name = link.name + s

    ## 翻转 inertial_origin 的 y 分量
    orig_parts = link.inertial_origin.split()
    y = float(orig_parts[1]) * sign
    if y == int(y):
        y = int(y)
    new.inertial_origin = f"{orig_parts[0]} {y} {' '.join(orig_parts[2:])}"

    ## 翻转 ixy, iyz
    new.ixy = link.ixy * sign
    new.iyz = link.iyz * sign

    ## 替换 mesh 前缀: left → right 或保持
    if sign == -1:
        new.mesh = link.mesh.replace("left", "right")
        # steer mesh: steerSupport_LF → steerSupport_RF 等
        mesh_map = {"LF": "RF", "LM": "RM", "LB": "RB"}
        for l, r in mesh_map.items():
            new.mesh = new.mesh.replace(f"steerSupport_{l}", f"steerSupport_{r}")
            new.mesh = new.mesh.replace(f"wheel_{l}", f"wheel_{r}")

    return new


def _add_ptz_mast(links, joints):
    # PTZ 桅杆: Support → Base → Yaw → Pitch → ZEDNav
    links.append(Link(
        name="PTZSupport_Link",
        inertial_origin="0 0 0.353", mass=0.687,
        ixx=0.0446, ixy=0, ixz=0, iyy=0.0445, iyz=0, izz=0.0004,
        mesh="PTZSupport_Link.STL", material="mast_mat",
    ))
    joints.append(Joint(
        name="mast_p_joint", jtype="fixed",
        parent="chassis", child="PTZSupport_Link",
        origin="0.372 0 0.356",
    ))

    links.append(Link(
        name="PTZBase_Link",
        inertial_origin="0 0 0.0015", mass=0.060,
        ixx=0.00006, ixy=0, ixz=0, iyy=0.00003, iyz=0, izz=0.00009,
        mesh="PTZBase_Link.STL", material="ptz_mat",
    ))
    joints.append(Joint(
        name="PTZBase_joint", jtype="fixed",
        parent="PTZSupport_Link", child="PTZBase_Link",
        origin="-0.00088 0 0.669",
    ))

    links.append(Link(
        name="PTZYaw_Link",
        inertial_origin="-0.383 0.027 -1.069", mass=1.0,
        ixx=0.298, ixy=0.0001, ixz=-0.015, iyy=0.254, iyz=0, izz=0.467,
        visual_origin="0.016 -0.027 0",
        mesh="PTZYaw_Link.STL", material="ptz_mat",
    ))
    joints.append(Joint(
        name="PTZYaw_joint", jtype="revolute",
        parent="PTZBase_Link", child="PTZYaw_Link",
        origin="0 0 0.023", axis="0 0 1",
        lower=-1.0, upper=1.0, effort=10, velocity=10,
        damping=0.2, friction=0.2,
    ))

    links.append(Link(
        name="PTZPitch_Link",
        inertial_origin="0 -0.06 0.03", mass=0.064,
        ixx=0.00007, ixy=0, ixz=0, iyy=0.00003, iyz=0, izz=0.0001,
        mesh="PTZPitch_Link.STL", material="ptz_mat",
    ))
    joints.append(Joint(
        name="PTZPitch_joint", jtype="revolute",
        parent="PTZYaw_Link", child="PTZPitch_Link",
        origin="0 0.055 0.062", origin_rpy="0 0.5 0", axis="0 1 0",
        lower=-1.0, upper=1.0, effort=10, velocity=10,
        damping=0.2, friction=0.2,
    ))

    links.append(Link(
        name="ZEDNav_Link",
        inertial_origin="0.0155 0 -0.00055", mass=0.3,
        ixx=0.0029, ixy=0, ixz=0, iyy=0.00018, iyz=0, izz=0.0029,
        mesh="ZEDNav_Link.STL", material="cam_mat",
        collision="mesh ZEDNav_Link.STL",
    ))
    joints.append(Joint(
        name="mast_cameras_joint", jtype="fixed",
        parent="PTZPitch_Link", child="ZEDNav_Link",
        origin="0 -0.055 0.047",
    ))


def _add_obstacle_cameras(links, joints):
    # 3 个障碍物相机（F/L/R），极小质量
    cam_inertia = dict(ixx=9.1875e-09, ixy=0.0, ixz=0.0, iyy=9.1875e-09, iyz=0.0, izz=1.378125e-08)

    links.append(Link(
        name="camera_obs_F_link", mass=0.001,
        mesh="obsZED_Link.STL", material="cam_mat",
        **cam_inertia,
    ))
    joints.append(Joint(
        name="obs_f_camera_joint", jtype="fixed",
        parent="chassis", child="camera_obs_F_link",
        origin="0.673358 0 0.313007", origin_rpy="0 0.4 0",
    ))

    links.append(Link(
        name="camera_obs_L_link", mass=0.001,
        **cam_inertia,
    ))
    joints.append(Joint(
        name="obs_l_camera_joint", jtype="fixed",
        parent="chassis", child="camera_obs_L_link",
        origin="0 0.462 0.321", origin_rpy="0.4 0 1.57",
    ))

    links.append(Link(
        name="camera_obs_R_link", mass=0.001,
        **cam_inertia,
    ))
    joints.append(Joint(
        name="obs_r_camera_joint", jtype="fixed",
        parent="chassis", child="camera_obs_R_link",
        origin="-0.006 -0.457 0.317", origin_rpy="-0.4 0 -1.57",
    ))


## ---- 主生成函数 ----

def generate_urdf():
    # 生成完整 URDF
    links, joints = [], []
    _add_chassis(links, joints)
    _add_wheel_assemblies(links, joints)
    _add_ptz_mast(links, joints)
    _add_obstacle_cameras(links, joints)

    parts = ['<?xml version="1.0"?>', '<robot name="zhurong_mars_rover">']

    # base_link（根节点，无对应 joint）
    parts.append(_fmt_link(links[0]))
    # base_c_joint + chassis（joint 在 child 之前）
    parts.append(_fmt_joint(joints[0]))
    parts.append(_fmt_link(links[1]))
    # 后续 links 和 joints 交替输出
    for i in range(2, len(links)):
        parts.append(_fmt_link(links[i]))
        parts.append(_fmt_joint(joints[i - 1]))

    parts.append("</robot>")
    return "\n".join(parts) + "\n"


def main():
    print("[INFO] Generating URDF...")
    urdf = generate_urdf()
    os.makedirs(os.path.dirname(OUTPUT_URDF), exist_ok=True)
    with open(OUTPUT_URDF, "w") as f:
        f.write(urdf)
    print(f"[DONE] {OUTPUT_URDF}")
    os.makedirs(ZHURONG_DESCRIPTION_URDF_DIR, exist_ok=True)
    with open(SYNCED_OUTPUT_URDF, "w") as f:
        f.write(urdf)
    print(f"[DONE] {SYNCED_OUTPUT_URDF}")


if __name__ == "__main__":
    main()
