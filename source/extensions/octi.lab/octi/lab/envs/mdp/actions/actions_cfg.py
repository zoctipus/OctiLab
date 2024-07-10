# Copyright (c) 2022-2024, The Octi and ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from octi.lab.envs.mdp.actions import task_space_actions
from omni.isaac.lab.utils import configclass

##
# Task-space Actions.
##


@configclass
class MultiConstraintsDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for inverse differential kinematics action term with multi constraints.
    This class amend attr body_name from type:str to type:list[str] reflecting its capability to
    received the desired positions, poses from multiple target bodies. This will be particularly
    useful for controlling dextrous hand robot with only positions of multiple key frame positions
    and poses, and output joint positions that satisfy key frame position/pose constrains

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    body_name: list[str] = MISSING

    class_type: type[ActionTerm] = task_space_actions.MultiConstraintDifferentialInverseKinematicsAction
