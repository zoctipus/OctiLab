# Copyright (c) 2022-2024, The Tycho Project Developers

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from . import actuator_pd
from omni.isaac.lab.actuators.actuator_cfg import ActuatorBaseCfg


@configclass
class HebiEffortMotorCfg(ActuatorBaseCfg):

    class_type: type = actuator_pd.HebiEffortMotor
    actuation_limit: list = MISSING


@configclass
class EffortMotorCfg(ActuatorBaseCfg):

    class_type: type = actuator_pd.EffortMotor


@configclass
class HebiStrategy3ActuatorCfg(ActuatorBaseCfg):
    """Configuration for MLP-based actuator model."""
    class_type: type = actuator_pd.HebiStrategy3Actuator
    '''
    We reference the stiffness(P) and damping(D)
    from gain_xml as how it done in tycho_master
    instead of setting them in stiffness and damping field below.
    '''

    position_p_scale: float = MISSING

    effort_p_scale: float = MISSING

    kp: list = MISSING

    ki: list = MISSING

    kd: list = MISSING

    i_clamp: list = MISSING

    min_target: list = MISSING

    max_target: list = MISSING

    target_lowpass: list = MISSING

    min_output: list = MISSING

    max_output: list = MISSING

    output_lowpass: list = MISSING

    d_on_error: list = MISSING

    maxtorque: list = MISSING

    speed_24v: list = MISSING


@configclass
class HebiStrategy4ActuatorCfg(HebiStrategy3ActuatorCfg):
    """Configuration for MLP-based actuator model."""
    class_type: type = actuator_pd.HebiStrategy4Actuator

    position_p_scale: float = MISSING

    effort_p_scale: float = MISSING

    p_p: list = MISSING

    p_d: list = MISSING

    e_p: list = MISSING

    e_d: list = MISSING


@configclass
class HebiDCMotorCfg(ActuatorBaseCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = actuator_pd.HebiDCMotor

    dt: float = MISSING

    p_p: list = MISSING

    p_d: list = MISSING

    e_p: list = MISSING

    e_d: list = MISSING

    saturation_effort: list = MISSING

    maxtorque: list = MISSING

    speed_24v: list = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""
