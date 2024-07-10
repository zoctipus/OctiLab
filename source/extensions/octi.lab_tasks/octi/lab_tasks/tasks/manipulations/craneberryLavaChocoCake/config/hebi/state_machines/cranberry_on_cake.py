# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a cabinet opening state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./orbit.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32

"""


"""Rest everything else."""

import torch
from omni.isaac.lab.assets import RigidObject
from collections.abc import Sequence
import warp as wp

from omni.isaac.lab.sensors import FrameTransformer
import octi.lab.envs.mdp as general_mdp


def simple_slerp(q0, q1, delta):
    # Normalize inputs
    q0 = q0 / torch.norm(q0, dim=1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=1, keepdim=True)

    # Compute the cosine of the angle between the quaternions
    dot = torch.sum(q0 * q1, dim=1, keepdim=True)
    # If the dot product is negative, slerp won't take the shorter path. Correct by reversing one quaternion.
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # Linearly interpolate and normalize the result
    result = (1 - delta) * q0 + delta * q1
    return result / torch.norm(result, dim=1, keepdim=True)


def move_towards(ee_pose, des_ee_pose, max_step):
    # Calculate the vector from current to desired position
    direction_vec = des_ee_pose[:, :3] - ee_pose[:, :3]
    distance = torch.norm(direction_vec, dim=1, keepdim=True)

    # Normalize the direction vector
    norm_direction_vec = direction_vec / distance.where(distance != 0, torch.ones_like(distance))

    # Calculate step size (either max_step or the remaining distance if it's smaller than max_step)
    step_size = torch.min(distance, torch.ones((distance.shape[0], 1), device=distance.device) * max_step)

    # Calculate new positions
    new_pos = ee_pose[:, :3] + norm_direction_vec * step_size

    # For orientation, it's tricky to apply the same logic directly,
    # so we'll just copy the desired orientation for now
    # This part requires proper quaternion interpolation (like SLERP) based on the "speed" or angular distance.
    new_orient = des_ee_pose[:, 3:]  # Placeholder, should use a proper interpolation method

    # Combine new position and orientation
    new_pose = torch.cat((new_pos, new_orient), dim=1)
    return new_pose


# initialize warp
wp.init()


class ChopState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class TychoCranberryPickingSmState:
    """States for the cabinet drawer opening state machine."""

    REST = wp.constant(0)
    APPROACH_CRANBERRY = wp.constant(1)
    GOTO_CRANBERRY = wp.constant(2)
    GRASP_CRANBERRY = wp.constant(3)
    APPROACH_ABOVE_CAKE = wp.constant(4)
    APPROACH_CAKE = wp.constant(5)
    RELEASE_CHOP = wp.constant(6)
    LIFT_HAND = wp.constant(7)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    cranberry_pose: wp.array(dtype=wp.transform),
    cake_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    cranberry_approach_offset: wp.array(dtype=wp.transform),
    cranberry_grasp_offset: wp.array(dtype=wp.transform),
    cake_approach_offset: wp.array(dtype=wp.transform),
    cranberry_drop_offset: wp.array(dtype=wp.transform),
    finish_lift_offset: wp.array(dtype=wp.transform),
    default_hand_quaternion: wp.array(dtype=wp.transform),
    tilted_hand_quaternion_wp: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == TychoCranberryPickingSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = ChopState.OPEN
        # wait for a while
    elif state == TychoCranberryPickingSmState.APPROACH_CRANBERRY:
        ee_pose_position = wp.transform_get_translation(cranberry_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(default_hand_quaternion[tid])
        # des_ee_pose[tid] = wp.transform(ee_pose_position, ee_pose_quaternion)
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cranberry_approach_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.APPROACH_CRANBERRY:
        #     # move to next state and reset wait time
        #     sm_state[tid] = TychoCranberryPickingSmState.GRASP_CRANBERRY
        #     sm_wait_time[tid] = 0.0
    elif state == TychoCranberryPickingSmState.GOTO_CRANBERRY:
        ee_pose_position = wp.transform_get_translation(cranberry_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(tilted_hand_quaternion_wp[tid])
        # des_ee_pose[tid] = wp.transform(ee_pose_position, ee_pose_quaternion)
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cranberry_grasp_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.OPEN
    elif state == TychoCranberryPickingSmState.GRASP_CRANBERRY:
        ee_pose_position = wp.transform_get_translation(cranberry_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(tilted_hand_quaternion_wp[tid])
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cranberry_grasp_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.CLOSE
        # wait for a while
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.GRASP_CRANBERRY:
        #     # move to next state and reset wait time
        #     sm_state[tid] = TychoCranberryPickingSmState.APPROACH_ABOVE_CAKE
        #     sm_wait_time[tid] = 0.0
    elif state == TychoCranberryPickingSmState.APPROACH_ABOVE_CAKE:
        ee_pose_position = wp.transform_get_translation(cake_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(default_hand_quaternion[tid])
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cake_approach_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.CLOSE
        # wait for a while
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.APPROACH_ABOVE_CAKE:
        #     # move to next state and reset wait time
        #     sm_state[tid] = TychoCranberryPickingSmState.APPROACH_CAKE
        #     sm_wait_time[tid] = 0.0
    elif state == TychoCranberryPickingSmState.APPROACH_CAKE:
        ee_pose_position = wp.transform_get_translation(cake_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(default_hand_quaternion[tid])
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cranberry_drop_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.CLOSE
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.APPROACH_CAKE:
        #     sm_state[tid] = TychoCranberryPickingSmState.RELEASE_CHOP
        #     sm_wait_time[tid] = 0.0
    elif state == TychoCranberryPickingSmState.RELEASE_CHOP:
        ee_pose_position = wp.transform_get_translation(cake_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(default_hand_quaternion[tid])
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(cranberry_drop_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.OPEN
        # wait for a while
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.RELEASE_CHOP:
        #     # move to next state and reset wait time
        #     sm_state[tid] = TychoCranberryPickingSmState.LIFT_HAND
        #     sm_wait_time[tid] = 0.0

    elif state == TychoCranberryPickingSmState.LIFT_HAND:
        ee_pose_position = wp.transform_get_translation(cake_pose[tid])
        ee_pose_quaternion = wp.transform_get_rotation(default_hand_quaternion[tid])
        _des_ee_pose = wp.transform(ee_pose_position, ee_pose_quaternion)
        des_ee_pose[tid] = wp.transform_multiply(finish_lift_offset[tid], _des_ee_pose)
        gripper_state[tid] = ChopState.OPEN
        # wait for a while
        # if sm_wait_time[tid] >= TychoCranberryPickingSmWaitTime.LIFT_HAND:
        #     # move to next state and reset wait time
        #     sm_state[tid] = TychoCranberryPickingSmState.LIFT_HAND
        #     sm_wait_time[tid] = 0.0
    # increment wait time
    # sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class CranberryDecoratorSm:
    """A simple state machine in a robot's task space to open a drawer in the cabinet.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_INFRONT_CRANBERRY: The robot moves towards the handle of the drawer.
    3. APPROACH_CRANBERRY: The robot grasps the handle of the drawer.
    4. APPROACH_CAKE: The robot opens the drawer.
    5. RELEASE_CHOP: The robot releases the handle of the drawer. This is the final state.
    """

    def __init__(self):
        pass

    def initialize(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.delta = 0.15
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # cranberry approach offset
        self.cranberry_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.cranberry_approach_offset[:, 0:3] = torch.tensor([0.05, -0.005, 0.01], device=self.device)
        self.cranberry_approach_offset[:, -1] = 1.0

        # cranberry grasp offset
        self.cranberry_grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.cranberry_grasp_offset[:, 0] = 0.0
        self.cranberry_grasp_offset[:, 1] = 0.0
        self.cranberry_grasp_offset[:, 2] = -0.00
        self.cranberry_grasp_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # approach infront of the cranberry
        self.cake_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.cake_approach_offset[:, 0:3] = torch.tensor([-0.0, -0.01, 0.155], device=self.device)
        self.cake_approach_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # cranberry grasp offset
        self.cranberry_drop_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.cranberry_drop_offset[:, 0:3] = torch.tensor([0.005, 0, 0.10194], device=self.device)
        self.cranberry_drop_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # approach infront of the cranberry
        self.finish_lift_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.finish_lift_offset[:, 0:3] = torch.tensor([0.001, -0.028, 0.165], device=self.device)
        self.finish_lift_offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        self.default_hand_quat = torch.zeros((self.num_envs, 7), device=self.device)
        self.tilted_hand_quat = torch.zeros((self.num_envs, 7), device=self.device)
        # warp expects quaternion as (x, y, z, w)
        # self.default_hand_quat[:] = torch.tensor([-0.4000, -0.3300,  0.0700, 0.0, 0.7292987, -0.6841955, 0.0], device = device)
        self.default_hand_quat[:] = torch.tensor(
            [-0.4000, -0.3300, 0.0700, -0.0126809, 0.7234, -0.6786616, 0.1262941], device=device
        )
        self.tilted_hand_quat[:] = torch.tensor(
            [-0.4000, -0.3300, 0.0700, -0.0126809, 0.7234, -0.6786616, 0.1262941], device=device
        )
        # self.default_hand_quat[:] = torch.tensor([-0.4000, -0.3300,  0.0700, 0.7071232, -0.7071232, 0, 0], device = device)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.cranberry_approach_offset_wp = wp.from_torch(self.cranberry_approach_offset, wp.transform)
        self.cranberry_grasp_offset_wp = wp.from_torch(self.cranberry_grasp_offset, wp.transform)
        self.cake_approach_offset_wp = wp.from_torch(self.cake_approach_offset, wp.transform)
        self.cranberry_drop_offset_wp = wp.from_torch(self.cranberry_drop_offset, wp.transform)
        self.finish_lift_offset_wp = wp.from_torch(self.finish_lift_offset, wp.transform)
        self.default_hand_quat_wp = wp.from_torch(self.default_hand_quat, wp.transform)
        self.tilted_hand_quat_wp = wp.from_torch(self.tilted_hand_quat, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        # reset state machine
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, env):
        cranberry: RigidObject = env.unwrapped.scene["canberry"]
        approach_cranberry_offset_tensor = torch.tensor([0.02, 0.0, -0.005], device=env.unwrapped.device)
        cake: RigidObject = env.unwrapped.scene["cake"]
        canberry_offset_tensor = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)
        above_cake_offset_tensor = torch.tensor([-0.0, -0.01, 0.15], device=env.unwrapped.device)
        ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]

        # -- end-effector frame
        tcp_rest_position = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
        tcp_rest_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()

        # -- cranberry frame
        cranberry_position = cranberry.data.body_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
        cranberry_orientation = cranberry.data.body_quat_w[..., 0, :].clone()

        cake_position = cake.data.body_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
        cake_orientation = cake.data.body_quat_w[..., 0, :].clone()

        cake_drop_position = env.unwrapped.data_manager.get_active_term("data", "cake_position_b")
        cranberry_eeframe_distance = env.unwrapped.data_manager.get_active_term("data", "canberry_eeframe_distance")
        cake_eeframe_distance = env.unwrapped.data_manager.get_active_term("data", "cake_eeframe_distance")
        canberry_cake_distance = env.unwrapped.data_manager.get_active_term("data", "canberry_cake_distance")
        canberry_above_cake_distance = general_mdp.get_body1_body2_distance(
            cranberry, cake, canberry_offset_tensor, above_cake_offset_tensor
        ).view(env.unwrapped.num_envs, -1)
        canberry_height = env.unwrapped.data_manager.get_active_term("data", "canberry_position")[:, 2].view(
            env.unwrapped.num_envs, -1
        )
        mid_chop_pos_b = env.unwrapped.data_manager.get_active_term("data", "chop_mid_position_b")
        mid_chop_pos_height = mid_chop_pos_b[:, 2].view(env.unwrapped.num_envs, -1)
        end_effector_speed = env.unwrapped.data_manager.get_active_term("data", "end_effector_speed")
        chop_pose = env.unwrapped.data_manager.get_active_term("data", "chop_pose")

        ee_berry_x_alignment = (torch.abs(tcp_rest_position[:, 0] - cranberry_position[:, 0]) < 0.0065).view(
            env.unwrapped.num_envs, -1
        )
        approach_cranberry_mask = ((canberry_height < 0.04) & (cranberry_eeframe_distance > 0.05)).view(-1)
        goto_cranberry_mask = ((canberry_height < 0.04) & (cranberry_eeframe_distance < 0.06)).view(-1)
        grasp_mask = (
            ee_berry_x_alignment
            & (cranberry_eeframe_distance < 0.008)
            & (torch.abs(canberry_height - mid_chop_pos_height) < 0.007)
        ).view(-1)
        approach_above_cake_mask = (
            (chop_pose < -0.45) & (canberry_above_cake_distance > 0.03) & (cranberry_eeframe_distance < 0.02)
        ).view(-1)
        approach_cake_mask = ((canberry_above_cake_distance <= 0.05) & (chop_pose < -0.44)).view(-1)
        release_mask = ((canberry_cake_distance <= 0.0125)).view(-1)
        lift_hand_mask = ((chop_pose > -0.38) & (canberry_cake_distance <= 0.022)).view(-1)
        # slow_ee_spped_mask = (end_effector_speed < 0.01).view(-1)

        self.sm_state[(self.sm_state == 6) & lift_hand_mask] = 7
        # self.sm_state[(self.sm_state == 4) & release_mask & slow_ee_spped_mask] = 5
        self.sm_state[(self.sm_state == 5) & release_mask] = 6
        self.sm_state[(self.sm_state == 4) & approach_cake_mask] = 5
        self.sm_state[(self.sm_state == 3) & approach_above_cake_mask] = 4
        # self.sm_state[(self.sm_state == 1) & grasp_mask & slow_ee_spped_mask] = 2
        self.sm_state[(self.sm_state == 2) & grasp_mask] = 3
        self.sm_state[(self.sm_state == 1) & goto_cranberry_mask] = 2
        self.sm_state[(self.sm_state == 0) & approach_cranberry_mask] = 1

        ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
        cranberry_pose = torch.cat([cranberry_position, cranberry_orientation], dim=-1)
        cake_pose = torch.cat([cake_position, cake_orientation], dim=-1)
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        cranberry_pose = cranberry_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        cake_pose = cake_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        cranberry_pose_wp = wp.from_torch(cranberry_pose.contiguous(), wp.transform)
        cake_pose_wp = wp.from_torch(cake_pose.contiguous(), wp.transform)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                cranberry_pose_wp,
                cake_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.cranberry_approach_offset_wp,
                self.cranberry_grasp_offset_wp,
                self.cake_approach_offset_wp,
                self.cranberry_drop_offset_wp,
                self.finish_lift_offset_wp,
                self.default_hand_quat_wp,
                self.tilted_hand_quat_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        delta_des_pose = move_towards(ee_pose, des_ee_pose, 0.015)
        # convert to torch
        return torch.cat([delta_des_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
