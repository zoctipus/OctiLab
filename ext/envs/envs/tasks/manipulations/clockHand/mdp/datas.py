import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.envs import ManagerBasedRLEnv
import ext.envs.cfgs.robots.hebi.mdp as tycho_mdp
import octilab.envs.mdp as general_mdp

def _from_eeframes_get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame):
    ee_tip_fixed_w = fixed_chop_tip_frame.data.target_pos_w[..., 0, :]
    ee_tip_free_w = free_chop_tip_frame.data.target_pos_w[..., 0, :]
    mid_pos_w = (ee_tip_fixed_w + ee_tip_free_w) / 2.0
    return mid_pos_w

def update_data(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    canberry_name: str = "canberry",
    cake_name: str = "cake",
    caneberry_offset: list[float] = [0.0, 0.0, 0.0],
    cake_offset: list[float] = [0.0, 0.0, 0.0],
    fixed_chop_frame_name: str = "frame_fixed_chop_tip",
    free_chop_frame_name: str = "frame_free_chop_tip",
)-> dict[str, torch.Tensor]:
    robot: Articulation = env.scene[robot_name]
    canberry: RigidObject = env.scene[canberry_name]
    canberry_scale = torch.tensor(canberry.cfg.spawn.scale, device = env.device)
    cake: RigidObject = env.scene[cake_name]
    cake_scale = torch.tensor(cake.cfg.spawn.scale, device = env.device)
    canberry_offset_tensor = torch.tensor(caneberry_offset, device = env.device) * canberry_scale
    cake_offset_tensor = torch.tensor(cake_offset, device = env.device) * cake_scale
    fixed_chop_frame:FrameTransformer = env.scene[fixed_chop_frame_name]
    free_chop_frame:FrameTransformer = env.scene[free_chop_frame_name]
    ee_frame:FrameTransformer = env.scene["ee_frame"]
    ee_frame_position_w = ee_frame.data.target_pos_w[..., 0, :3]
    
    canberry_position = canberry.data.root_pos_w + canberry_offset_tensor
    cake_position = cake.data.root_pos_w + cake_offset_tensor
    chop_mid_position_w = ee_frame_position_w 
    end_effector_speed = torch.norm(robot.data.body_vel_w[..., 7, :3], dim=1).view(-1, 1)
    canberry_eeframe_distance = torch.norm(canberry_position - chop_mid_position_w, dim=1).view(-1, 1)
    cake_eeframe_distance = torch.norm(cake_position - chop_mid_position_w, dim=1).view(-1, 1)
    canberry_cake_distance = torch.norm(cake_position - canberry_position, dim=1).view(-1, 1)
    canberry_height = canberry_position[:, 2].view(-1, 1)
    chop_pose = robot.data.joint_pos[:, -1].view(-1, 1)
    
    chop_mid_position_b, _ = canberry_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], chop_mid_position_w
    )

    canberry_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], canberry_position
    )

    cake_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cake_position
    )


    # using the the fact that consine of two vectors angle greater than 90 degree is negative,
    # and the two vectors angle less than 90 degree is positive
    # in this case vector1 is ball to fixedchoptip, vector 2 is ball to freechoptip
    # if cos_angle v1 and v2 is less than -5 it mean chop is around ball, a good indicator for close chop
    vec_to_fix_tip = fixed_chop_frame.data.target_pos_w[..., 0, :3] - canberry_position
    vec_to_rot_tip = free_chop_frame.data.target_pos_w[..., 0, :3] - canberry_position
    norm_vec_to_left_tip = vec_to_fix_tip / torch.norm(vec_to_fix_tip, dim=1, keepdim=True)
    norm_vec_to_right_tip = vec_to_rot_tip / torch.norm(vec_to_rot_tip, dim=1, keepdim=True)
    cos_angle = torch.einsum('ij,ij->i', norm_vec_to_left_tip, norm_vec_to_right_tip).view(-1, 1)

    canberry_grasp_mask = (chop_pose < -0.4) & (cos_angle < -0.5) & (canberry_eeframe_distance < 0.02)
    dict = {"canberry_position" : canberry_position,
            "cake_position" : cake_position,
            "canberry_position_b" : canberry_position_b,
            "cake_position_b" : cake_position_b,
            "chop_mid_position_w" : chop_mid_position_w,
            "chop_mid_position_b" : chop_mid_position_b,
            "end_effector_speed" : end_effector_speed,
            "canberry_eeframe_distance" : canberry_eeframe_distance,
            "cake_eeframe_distance" : cake_eeframe_distance,
            "canberry_cake_distance" : canberry_cake_distance,
            "canberry_height" : canberry_height,
            "chop_pose" : chop_pose,
            "chop_tips_canberry_cos_angle" : cos_angle,
            "canberry_grasp_mask": canberry_grasp_mask}
    return dict


def update_history(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
) -> dict[str, torch.Tensor]:
    robot: Articulation = env.scene[robot_name]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    ee_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_frame.data.target_pos_w[..., 0, :3]
    )
    end_effector_speed = torch.norm(robot.data.body_vel_w[..., 7, :3], dim=1).view(-1, 1)

    dict = {"end_effector_speed" : end_effector_speed,
            "ee_position_b": ee_position_b}
    return dict
