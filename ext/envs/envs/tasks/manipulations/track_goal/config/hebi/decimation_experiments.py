# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from .Hebi_JointPos_GoalTracking_Env import IdealPDHebi_JointPos_GoalTracking_Env


class IdealPDHebi_JointPos_GoalTracking_Decimate1_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 1
        self.sim.dt = agent_update_hz / self.decimation


class IdealPDHebi_JointPos_GoalTracking_Decimate2_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 2
        self.sim.dt = agent_update_hz / self.decimation


class IdealPDHebi_JointPos_GoalTracking_Decimate5_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 5
        self.sim.dt = agent_update_hz / self.decimation
