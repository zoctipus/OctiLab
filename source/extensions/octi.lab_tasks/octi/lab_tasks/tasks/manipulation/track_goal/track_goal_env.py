# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs.mdp import time_out


class ObjectSceneCfg(InteractiveSceneCfg):

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class RewardsCfg:
    pass


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass


@configclass
class EventCfg:
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass
