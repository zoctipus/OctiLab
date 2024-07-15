# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.kit.commands
from pxr import Gf, Sdf, Usd
import random
import re
from pxr import Semantics, UsdGeom, Vt
from omni.isaac.lab.sim.utils import find_matching_prim_paths
from omni.isaac.lab.sim.schemas import activate_contact_sensors
if TYPE_CHECKING:
    from . import from_files_cfg


def spawn_multi_object_randomly_sdf(
    prim_path: str,
    cfg: from_files_cfg.MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # spawn everything first in a "Dataset" prim
    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Object_{index:02d}"
        prim = asset_cfg.func(proto_prim_path, asset_cfg)
        # save the proto prim path
        proto_prim_paths.append(proto_prim_path)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            activate_contact_sensors(proto_prim_path, asset_cfg.activate_contact_sensors)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # convert orientation ordering (wxyz to xyzw)
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            # randomly select an asset configuration
            proto_path = random.choice(proto_prim_paths)
            # inherit the proto prim
            # env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*translation)

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatd(*orientation)

            scale_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            if cfg.uniform_scale:
                upper_bound = min(cfg.scaling_range['x'][1], cfg.scaling_range['y'][1], cfg.scaling_range['z'][1])
                lower_bound = max(cfg.scaling_range['x'][0], cfg.scaling_range['y'][0], cfg.scaling_range['z'][0])
                x = y = z = random.choice((lower_bound, upper_bound))
            else:
                x = random.choice(cfg.scaling_range['x'])
                y = random.choice(cfg.scaling_range['y'])
                z = random.choice(cfg.scaling_range['z'])
            scale_spec.default = Gf.Vec3d(x, y, z)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = env_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())

    # for i, prim_path in enumerate(prim_paths):
    #     prim = stage.GetPrimAtPath(prim_path)
    #     if not UsdPhysics.CollisionAPI(prim):
    #         print('hi')
    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_object_randomly(
    prim_path: str,
    cfg: from_files_cfg.MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for prim_path in prim_paths:
        # randomly select an asset configuration
        asset_cfg = random.choice(cfg.assets_cfg)
        # spawn single instance
        prim = asset_cfg.func(prim_path, asset_cfg, translation, orientation)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # return the prim
    return prim
