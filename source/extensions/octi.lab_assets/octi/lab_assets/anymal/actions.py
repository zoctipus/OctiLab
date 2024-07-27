from __future__ import annotations
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (JointPositionActionCfg, JointEffortActionCfg)
"""
LEAP ACTIONS
"""

ANYMAL_C_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

ANYMAL_C_JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=20, debug_vis=True)