"""USD 写入 + RViz DAE 导出:mesh/分层地形/岩石 USD + 贴图/物理材质 + DAE(pycollada 隔离在此)。"""

import os

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Gf, UsdPhysics, UsdShade
from collada import Collada, material as collada_material, scene as collada_scene, geometry as collada_geometry
from collada.source import FloatSource, InputList
from collada.material import CImage, Surface, Sampler2D, Map

from .terrain import _split_triangles_by_height_layers


# ── USD mesh 写入 + 材质 ──


def _define_usd_mesh_prim(stage, mesh_path, vertices, faces, uvs=None):
    # 创建 UsdGeom.Mesh 并设置基础属性：顶点/面/细分/Extent/UV
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)

    pts = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    mesh_prim.GetPointsAttr().Set(pts)
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(faces))
    mesh_prim.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())
    mesh_prim.GetSubdivisionSchemeAttr().Set("none")

    ## UV 坐标（UsdGeomPrimvarsAPI）
    if uvs is not None:
        pv_api = UsdGeom.PrimvarsAPI(mesh_prim.GetPrim())
        uv_primvar = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv_primvar.Set([Gf.Vec2f(float(u), float(v)) for u, v in uvs])

    ## Extent（包围盒）
    arr = np.asarray(vertices)
    mesh_prim.GetExtentAttr().Set([
        Gf.Vec3f(float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].min())),
        Gf.Vec3f(float(arr[:, 0].max()), float(arr[:, 1].max()), float(arr[:, 2].max())),
    ])

    return mesh_prim


def _add_texture_material(stage, mesh_prim_path, texture_path, normal_map_path=None):
    # 给 mesh 添加 UsdShade 贴图材质（UsdPreviewSurface + 贴图 reader）
    abs_tex = os.path.abspath(texture_path) if texture_path else ""
    abs_nrm = os.path.abspath(normal_map_path) if normal_map_path else ""

    material_path = f"{mesh_prim_path}/TerrainMaterial"
    material = UsdShade.Material.Define(stage, material_path)

    # PreviewSurface shader
    surf = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

    # UV reader
    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    if abs_tex and os.path.exists(abs_tex):
        tex = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(abs_tex)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        print(f"[INFO] Applied texture: {abs_tex}")
    else:
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.4, 0.3))

    if abs_nrm and os.path.exists(abs_nrm):
        nrm = UsdShade.Shader.Define(stage, f"{material_path}/NormalTex")
        nrm.CreateIdAttr("UsdUVTexture")
        nrm.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(abs_nrm)
        nrm.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        nrm.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
        nrm.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
        nrm.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        nrm.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        nrm.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        nrm.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm.ConnectableAPI(), "rgb")
        print(f"[INFO] Applied normal map: {abs_nrm}")

    # 绑定到 mesh
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)


def _bind_physics_material(stage, prim_path, material_name, material_cfg):
    # 创建并绑定物理材质（摩擦/弹性）到碰撞体
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    material_path = f"{prim_path}/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(float(material_cfg["static_friction"]))
    mat_api.CreateDynamicFrictionAttr().Set(float(material_cfg["dynamic_friction"]))
    mat_api.CreateRestitutionAttr().Set(float(material_cfg.get("restitution", 0.0)))

    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return True


def _safe_prim_name(name):
    """Convert a config name into a conservative USD prim name."""
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return safe or "layer"


def _add_semantic_label(prim, label):
    """Author a persistent Replicator semantic class label."""
    label = _safe_prim_name(label)
    prim.AddAppliedSchema("SemanticsLabelsAPI:class")
    prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set([label])


def write_mesh_usd(output_path, vertices, triangles, add_collision=True, root_name="ground",
                   texture_path=None, normal_map_path=None, uvs=None, physics_material_cfg=None):
    # 将三角形 mesh 写入 USD 文件（default prim + 子 mesh + 可选碰撞/贴图）
    stage = Usd.Stage.CreateNew(output_path)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())

    mesh_path = f"/{root_name}/mesh"
    mesh_prim = _define_usd_mesh_prim(stage, mesh_path, vertices, triangles, uvs)

    if add_collision:
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        if physics_material_cfg is not None:
            _bind_physics_material(stage, mesh_path, "physicsMaterial", physics_material_cfg)

    if texture_path:
        try:
            _add_texture_material(stage, mesh_path, texture_path, normal_map_path)
        except Exception as e:
            print(f"[WARN] Failed to apply texture: {e}")

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {output_path} ({len(vertices)} vertices, {len(triangles)} triangles)")


def write_layered_terrain_usd(output_path, vertices, triangles, layer_cfgs, root_name="terrain",
                              texture_path=None, normal_map_path=None, uvs=None, add_collision=True,
                              vertex_field=None):
    # 写分层地形 USD：按高度或显式层场拆分，每层一个 mesh prim。
    stage = Usd.Stage.CreateNew(output_path)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())

    layers = _split_triangles_by_height_layers(vertices, triangles, layer_cfgs, vertex_field=vertex_field)
    total_tris = 0
    for layer_name, layer_triangles, layer_cfg in layers:
        semantic_label = layer_cfg.get("semantic_label", layer_name)
        layer_name = _safe_prim_name(layer_name)

        ## 提取子集顶点，重映射三角形索引
        used_vertex_ids = np.unique(layer_triangles.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used_vertex_ids] = np.arange(len(used_vertex_ids))
        layer_vertices = vertices[used_vertex_ids]
        layer_faces = remap[layer_triangles]
        layer_uvs = uvs[used_vertex_ids] if uvs is not None else None

        mesh_path = f"/{root_name}/{layer_name}"
        mesh_prim = _define_usd_mesh_prim(stage, mesh_path, layer_vertices, layer_faces, layer_uvs)
        _add_semantic_label(mesh_prim.GetPrim(), semantic_label)

        if add_collision:
            UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
            _bind_physics_material(stage, mesh_path, "physicsMaterial", layer_cfg)

        layer_texture_path = layer_cfg.get("texture_path", texture_path)
        if layer_texture_path:
            try:
                _add_texture_material(stage, mesh_path, layer_texture_path, normal_map_path)
            except Exception as e:
                print(f"[WARN] Failed to apply texture on {layer_name}: {e}")

        total_tris += len(layer_triangles)

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {output_path} ({len(layers)} terrain layers, {total_tris} triangles)")


def write_rocks_usd(output_path, rock_groups, add_collision=True, root_name="obstacles"):
    # 将多组石头（每组独立贴图）写入单个 USD
    # rock_groups: [(trimesh, texture_abs_path_or_None, rock_idx), ...]
    stage = Usd.Stage.CreateNew(output_path)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())

    total_v, total_f = 0, 0
    for mesh, tex_path, rock_idx in rock_groups:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if len(verts) == 0 or len(faces) == 0:
            continue

        grp_path = f"/{root_name}/rock_{rock_idx}"
        UsdGeom.Xform.Define(stage, grp_path)
        mesh_path = f"{grp_path}/mesh"

        ## UV：优先用 trimesh visual.uv，否则用 XY 平面投影
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        if uv is None or len(uv) != len(verts):
            xy = verts[:, :2]
            xy_min = xy.min(axis=0)
            xy_span = np.maximum(xy.max(axis=0) - xy_min, 1.0e-6)
            uv = (xy - xy_min) / xy_span

        mesh_prim = _define_usd_mesh_prim(stage, mesh_path, verts, faces, uv)
        _add_semantic_label(mesh_prim.GetPrim(), "rock")

        if add_collision:
            UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())

        # 绑贴图
        if tex_path:
            try:
                _add_texture_material(stage, mesh_path, tex_path)
            except Exception as e:
                print(f"[WARN] rock_{rock_idx} texture failed: {e}")

        total_v += len(verts)
        total_f += len(faces)

    stage.GetRootLayer().Save()
    print(f"[INFO] Saved {output_path} ({len(rock_groups)} groups, {total_v} vertices, {total_f} triangles)")


# ── RViz DAE 导出(pycollada)──


def _export_dae_with_texture(output_dae, vertices, faces, uvs, texture_path):
    # 用 pycollada 导出带贴图的 DAE（供 RViz 可视化）

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if uvs is None:
        print("[WARN] No UV provided, using XZ projection")
        uvs = vertices[:, [0, 2]]
        uvs = (uvs - uvs.min()) / (uvs.max() - uvs.min() + 1e-6)
    else:
        uvs = np.asarray(uvs, dtype=np.float32)

    # 创建 COLLADA
    dae = Collada()

    tex_name = os.path.basename(texture_path)
    img = CImage("terrain_img", tex_name, dae)
    dae.images.append(img)

    surface = Surface("terrain_surface", img)
    sampler = Sampler2D("terrain_sampler", surface)
    effect = collada_material.Effect("terrain_effect", [surface, sampler], "lambert",
                             diffuse=Map(sampler, "UVMap"))
    mat = collada_material.Material("terrain_material", "terrain", effect)
    dae.effects.append(effect)
    dae.materials.append(mat)

    vert_src = FloatSource("vertices", vertices.flatten(), ('X', 'Y', 'Z'))
    uv_src = FloatSource("uvs", uvs.flatten(), ('S', 'T'))
    geom = collada_geometry.Geometry(dae, "terrain_geom", "terrain", [vert_src, uv_src])

    input_list = InputList()
    input_list.addInput(0, 'VERTEX', "#vertices")
    input_list.addInput(1, 'TEXCOORD', "#uvs")

    # COLLADA triangles 需要交错索引: [v0, uv0, v1, uv1, v2, uv2, ...]
    # 对于 vertex interpolation，UV 索引 = 顶点索引
    flat_faces = faces.reshape(-1)
    indices = np.empty(len(flat_faces) * 2, dtype=np.int32)
    indices[0::2] = flat_faces  # vertex index
    indices[1::2] = flat_faces  # uv index (same for vertex interpolation)

    triset = geom.createTriangleSet(indices, input_list, "terrain_material")
    geom.primitives.append(triset)
    dae.geometries.append(geom)

    # GeometryNode 需要 MaterialNode，不是 Material
    mat_node = collada_scene.MaterialNode("terrain_material", mat, [])
    geom_node = collada_scene.GeometryNode(geom, [mat_node])
    node = collada_scene.Node("terrain_node", children=[geom_node])
    myscene = collada_scene.Scene("scene", [node])
    dae.scenes.append(myscene)
    dae.scene = myscene

    dae.write(output_dae)
    print(f"[INFO] Saved {output_dae} ({len(vertices)} verts, with texture {tex_name})")
