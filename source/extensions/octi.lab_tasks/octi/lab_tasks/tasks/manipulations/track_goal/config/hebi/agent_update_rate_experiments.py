# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from .Hebi_JointPos_GoalTracking_Env import IdealPDHebi_JointPos_GoalTracking_Env


class IdealPDHebi_JointPos_GoalTracking_Agent4Hz_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.05
        self.decimation = 5


class IdealPDHebi_JointPos_GoalTracking_Agent6dot25Hz_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.04
        self.decimation = 4


class IdealPDHebi_JointPos_GoalTracking_Agent11dot11Hz_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.03
        self.decimation = 3


class IdealPDHebi_JointPos_GoalTracking_Agent25Hz_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.02
        self.decimation = 2


class IdealPDHebi_JointPos_GoalTracking_Agent100Hz_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.01
        self.decimation = 1
