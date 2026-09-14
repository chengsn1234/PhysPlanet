#!/usr/bin/env python3
"""祝融号 USD 材质绑定脚本

IsaacLab convert_urdf 使用 instancing + variant payload 分层结构：
- zhurong.usd: 顶层，variant sets + payload 引用
- configuration/zhurong_base.usd: mesh 数据，instancing source
- configuration/zhurong_physics.usd: 物理属性
- configuration/zhurong_robot.usd: 关节

材质必须在 zhurong_base.usd 里创建和绑定（mesh 实际存放在这个文件中）。

用法:
    conda activate env_isaaclab
    python source/physplanet_assets/tools/bind_zhurong_materials.py
"""
import os
from pxr import Usd, UsdShade, Sdf, Gf, UsdGeom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "physplanet_assets", "zhurong_assets")
BASE_USD = os.path.join(ASSETS_DIR, "configuration", "zhurong_base.usd")

MATERIAL_COLORS = {
    'chassis_mat': (0.51, 0.38, 0.15),
    'arm_mat': (0.79, 0.82, 0.93),
    'steer_mat': (0.79, 0.82, 0.93),
    'wheel_mat': (0.53, 0.51, 0.49),
    'mast_mat': (0.51, 0.38, 0.15),
    'ptz_mat': (0.79, 0.82, 0.93),
    'cam_mat': (1.0, 0.74, 0.59),
}

# mesh 路径关键字 -> 材质名
LINK_TO_MATERIAL = {
    'chassis': 'chassis_mat',
    'base_link': 'chassis_mat',
    'front_wheel': 'wheel_mat',
    'middle_wheel': 'wheel_mat',
    'back_wheel': 'wheel_mat',
    'wheel_LF': 'wheel_mat',
    'wheel_LM': 'wheel_mat',
    'wheel_LB': 'wheel_mat',
    'wheel_RF': 'wheel_mat',
    'wheel_RM': 'wheel_mat',
    'wheel_RB': 'wheel_mat',
    'steerSupport': 'steer_mat',
    'suspension_arm': 'arm_mat',
    'suspension_steer': 'steer_mat',
    'leftMainRocker': 'arm_mat',
    'rightMainRocker': 'arm_mat',
    'leftSubRocker': 'arm_mat',
    'rightSubRocker': 'arm_mat',
    'PTZSupport': 'mast_mat',
    'PTZBase': 'ptz_mat',
    'PTZYaw': 'ptz_mat',
    'PTZPitch': 'ptz_mat',
    'ZEDNav': 'cam_mat',
    'camera_obs': 'cam_mat',
}


def create_material(stage, looks_path, mat_name, color):
    mat_path = f'{looks_path}/{mat_name}'
    mat_prim = stage.DefinePrim(mat_path, 'Material')
    material = UsdShade.Material(mat_prim)
    shader_path = f'{mat_path}/Shader'
    shader = UsdShade.Shader(stage.DefinePrim(shader_path, 'Shader'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def main():
    base_path = os.path.normpath(BASE_USD)
    stage = Usd.Stage.Open(base_path)
    looks_path = '/zhurong_mars_rover/Looks'

    # 创建材质
    for mat_name, color in MATERIAL_COLORS.items():
        create_material(stage, looks_path, mat_name, color)
        print(f'  Created {mat_name}')

    # 直接在 mesh prim 上绑定材质
    bound = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        mat_name = 'arm_mat'
        for key, value in LINK_TO_MATERIAL.items():
            if key in path:
                mat_name = value
                break
        material = UsdShade.Material(stage.GetPrimAtPath(f'{looks_path}/{mat_name}'))
        if material:
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            bound += 1
            print(f'  {path} -> {mat_name}')

    stage.GetRootLayer().Save()
    print(f'\n[DONE] {bound} meshes bound, saved to {base_path}')


if __name__ == '__main__':
    main()
