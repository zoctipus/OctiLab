# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from ... import fetching_env
##
# Pre-defined configs
##
import octi.lab_assets.anymal as anymal

@configclass
class ActionsCfg:
    actions = anymal.ANYMAL_C_JOINT_POSITION

@configclass
class AnymalCRoughPositionEnvCfg(fetching_env.LocomotionFetchingRoughEnvCfg):
    actions:ActionsCfg = ActionsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
