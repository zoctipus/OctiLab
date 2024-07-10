# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


@configclass
class OctiManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    datas: object = MISSING
