# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from .Hebi_JointPos_GoalTracking_Env import IdealPDHebi_JointPos_GoalTracking_Env


class IdealPDHebi_JointPos_GoalTracking_0dot5p_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.stiffness *= 0.5


class IdealPDHebi_JointPos_GoalTracking_1dot5p_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.stiffness *= 1.5


class IdealPDHebi_JointPos_GoalTracking_2p_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 2  # type: ignore


class IdealPDHebi_JointPos_GoalTracking_5p_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.stiffness *= 5


class IdealPDHebi_JointPos_GoalTracking_10p_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.stiffness *= 10


class IdealPDHebi_JointPos_GoalTracking_0dot2d_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.damping *= 0.2


class IdealPDHebi_JointPos_GoalTracking_0dot5d_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.damping *= 0.5


class IdealPDHebi_JointPos_GoalTracking_1dot5d_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.damping *= 1.5


class IdealPDHebi_JointPos_GoalTracking_2d_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.damping *= 2


class IdealPDHebi_JointPos_GoalTracking_5p_0dot5d_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        for key, actuator in self.scene.robot.actuators.items():  # type: ignore
            actuator.stiffness *= 5
            actuator.damping *= 0.5
