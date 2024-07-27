# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.actuators.actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import HebiStrategy3ActuatorCfg, HebiEffortMotorCfg, EffortMotorCfg, HebiStrategy4ActuatorCfg, HebiDCMotorCfg


class EffortMotor(ActuatorBase):
    cfg: EffortMotorCfg

    def __init__(self, cfg: EffortMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        self.computed_effort = torch.clip(control_action.joint_efforts, -self.effort_limit, self.effort_limit)
        self.applied_effort = self.computed_effort
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def reset(self, env_ids: Sequence[int]):
        pass


class HebiEffortMotor(ActuatorBase):
    cfg: HebiEffortMotorCfg

    def __init__(self, cfg: HebiEffortMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.actuation_limit = torch.tensor(cfg.actuation_limit, device=self._device)

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        self.computed_effort = control_action.joint_efforts * self.actuation_limit
        self.applied_effort = self.computed_effort
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def reset(self, env_ids: Sequence[int]):
        pass


class HebiStrategy3Actuator(ActuatorBase):

    cfg: HebiStrategy3ActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HebiStrategy3ActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.num_envs = kwargs['num_envs']
        self.device = kwargs['device']
        self.position_p_scale = torch.tensor(cfg.position_p_scale, device=self.device)
        self.effort_p_scale = torch.tensor(cfg.effort_p_scale, device=self.device)
        self.kp = torch.tensor(cfg.kp, device=self.device)
        self.kp[0] *= self.position_p_scale
        self.kp[2] *= self.effort_p_scale
        self.kp = self.kp.view(-1)
        self.ki = torch.tensor(cfg.ki, device=self.device).view(-1)
        self.kd = torch.tensor(cfg.kd, device=self.device).view(-1)
        self.i_clamp = torch.tensor(cfg.i_clamp, device=self.device).view(-1)
        self.min_target = torch.tensor(cfg.min_target, device=self.device).view(-1)
        self.max_target = torch.tensor(cfg.max_target, device=self.device).view(-1)
        self.target_lowpass = torch.tensor(cfg.target_lowpass, device=self.device).view(-1)
        self.min_output = torch.tensor(cfg.min_output, device=self.device).view(-1)
        self.max_output = torch.tensor(cfg.max_output, device=self.device).view(-1)
        self.output_lowpass = torch.tensor(cfg.output_lowpass, device=self.device).view(-1)
        self.d_on_error = torch.tensor(cfg.d_on_error, device=self.device).view(-1)
        self.maxtorque = torch.tensor(cfg.maxtorque, device=self.device).view(-1)
        self.speed_24v = torch.tensor(cfg.speed_24v, device=self.device).view(-1)

        self.errSum = torch.zeros((self.num_envs, 21), device=self.device)
        self.lastDOn = torch.zeros((self.num_envs, 21), device=self.device)
        self.last_target = torch.zeros((self.num_envs, 21), device=self.device)
        self.last_output = torch.zeros((self.num_envs, 21), device=self.device)

    """
    Operations.
    """
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_effort: torch.Tensor
    ) -> ArticulationActions:
        joint_des = torch.cat([control_action.joint_positions,
                               control_action.joint_velocities,
                               control_action.joint_efforts], dim=1)
        joint = torch.cat([joint_pos, joint_vel, joint_effort], dim=1)
        joint_des = torch.clip(joint_des, self.min_target, self.max_target)
        # broadcasting rule is applied at | here
        target = torch.where((self.last_target == 0) | (self.target_lowpass == 1),
                             joint_des, self.last_target + self.target_lowpass * (joint_des - self.last_target))
        self.last_target = target
        error = target - joint
        self.errSum += error

        dErr = torch.where(self.lastDOn == 1, error - self.lastDOn, self.lastDOn - joint)
        self.lastDOn = torch.where(self.lastDOn == 1, error, joint)
        i_output = torch.clip(self.errSum * self.ki, -self.i_clamp, self.i_clamp)
        output = self.kp * error + i_output + self.kd * dErr
        output = torch.clip(output, self.min_output, self.max_output)
        # broadcasting rule is applied at | here
        output = torch.where((self.output_lowpass == 1),
                             output, self.last_output + self.output_lowpass * (output - self.last_output))
        self.last_output = output
        pwm = torch.sum(output.view(self.num_envs, 3, -1), dim=1)
        torque_des = self.pwm_to_torque_(pwm, joint_vel)

        # setting below field passed to "data" for record purpose
        self.computed_effort = torque_des
        self.applied_effort = torque_des

        # control_action.joint_efforts is the field will be passed to simulator for physic stepping
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def pwm_to_torque_(self, pwm, joint_vel):
        ctrl = torch.multiply(pwm - torch.divide(torch.abs(joint_vel), self.speed_24v), self.maxtorque)
        ctrl = torch.clip(ctrl, -self.maxtorque, self.maxtorque)
        return ctrl

    def reset(self, env_ids: Sequence[int]):
        self.errSum[env_ids] = 0
        self.lastDOn[env_ids] = 0
        self.last_target[env_ids] = 0
        self.last_output[env_ids] = 0


class HebiStrategy4Actuator(ActuatorBase):

    cfg: HebiStrategy4ActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HebiStrategy4ActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.num_envs = kwargs['num_envs']
        self.device = kwargs['device']
        self.position_p_scale = torch.tensor(cfg.position_p_scale, device=self.device)
        self.effort_p_scale = torch.tensor(cfg.effort_p_scale, device=self.device)
        self.p_p = torch.tensor(cfg.p_p, device=self.device).view(-1) * self.position_p_scale
        self.p_d = torch.tensor(cfg.p_d, device=self.device).view(-1)
        self.e_p = torch.tensor(cfg.e_p, device=self.device).view(-1) * self.effort_p_scale
        self.e_d = torch.tensor(cfg.e_d, device=self.device).view(-1)
        self.i_clamp = torch.tensor(cfg.i_clamp, device=self.device).view(-1)
        self.min_target = torch.tensor(cfg.min_target, device=self.device).view(-1)
        self.max_target = torch.tensor(cfg.max_target, device=self.device).view(-1)
        self.target_lowpass = torch.tensor(cfg.target_lowpass, device=self.device).view(-1)
        self.min_output = torch.tensor(cfg.min_output, device=self.device).view(-1)
        self.max_output = torch.tensor(cfg.max_output, device=self.device).view(-1)
        self.output_lowpass = torch.tensor(cfg.output_lowpass, device=self.device).view(-1)
        self.d_on_error = torch.tensor(cfg.d_on_error, device=self.device).view(-1)
        self.maxtorque = torch.tensor(cfg.maxtorque, device=self.device).view(-1)
        self.speed_24v = torch.tensor(cfg.speed_24v, device=self.device).view(-1)
        self.errSum = torch.zeros((self.num_envs, 21), device=self.device)
        self.lastDOn = torch.zeros((self.num_envs, 21), device=self.device)
        self.last_target = torch.zeros((self.num_envs, 21), device=self.device)
        self.last_output = torch.zeros((self.num_envs, 21), device=self.device)
        self.last_effort = torch.zeros((self.num_envs, 7), device=self.device)

    """
    Operations.
    """
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_effort: torch.Tensor
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("Hebi pwm motor requires a desired(target) joint position to calculate the \
                            the joint force. You must use action that outputs joint position.")

        else:
            error_pos = control_action.joint_positions - joint_pos
            target_eff = self.p_p * error_pos + self.p_d * joint_vel + control_action.joint_efforts

            error_eff = target_eff - joint_effort
            impulse = target_eff - self.last_effort
            pwm = self.e_p * error_eff + self.e_d * impulse
            pwm = torch.clip(pwm, -1, 1)
            torque_des = self.pwm_to_torque_(target_eff, joint_vel)
            self.last_effort[:] = torque_des

            # setting below field passed to "data" for record purpose
            self.computed_effort = torque_des
            self.applied_effort = torque_des

            # control_action.joint_efforts is the field will be passed to simulator for physic stepping
            control_action.joint_efforts = self.applied_effort
            control_action.joint_positions = None
            control_action.joint_velocities = None
            return control_action

    def reset(self, env_ids: Sequence[int]):
        self.errSum[env_ids] = 0
        self.lastDOn[env_ids] = 0
        self.last_target[env_ids] = 0
        self.last_output[env_ids] = 0

    def pwm_to_torque_(self, pwm, joint_vel):
        ctrl = torch.multiply(pwm - torch.divide(torch.abs(joint_vel), self.speed_24v), self.maxtorque)
        ctrl = torch.clip(ctrl, -self.maxtorque, self.maxtorque)
        return ctrl


class HebiDCMotor(ActuatorBase):

    cfg: HebiDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HebiDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.p_p = torch.tensor(cfg.p_p, device=self._device).view(-1)
        self.p_d = torch.tensor(cfg.p_d, device=self._device).view(-1)
        self.e_p = torch.tensor(cfg.e_p, device=self._device).view(-1)
        self.e_d = torch.tensor(cfg.e_d, device=self._device).view(-1)
        self.speed_24v = torch.tensor(cfg.speed_24v, device=self._device).view(-1)
        self.saturation_effort = torch.tensor(cfg.saturation_effort, device=self._device).view(-1)
        self.maxtorque = torch.tensor(cfg.maxtorque, device=self._device).view(-1)
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        # calculate the desired joint torques
        self.computed_effort = self.p_p * 5 * error_pos + self.p_d * joint_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits
        # -- max limit
        max_effort = self.saturation_effort * (1.0 - self._joint_vel / self.speed_24v)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.maxtorque)
        # -- min limit
        min_effort = self.saturation_effort * (-1.0 - self._joint_vel / self.speed_24v)
        min_effort = torch.clip(min_effort, min=-self.maxtorque, max=self._zeros_effort)

        # clip the torques based on the motor limits
        return torch.clip(effort, min=min_effort, max=max_effort)

    def reset(self, env_ids: Sequence[int]):
        pass
