from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.assets import RigidObject, Articulation
from octilab.assets import Deformable
import h5py
from octilab.envs import HebiRLTaskEnv


def record_state_configuration(
    env: HebiRLTaskEnv,
    env_ids: torch.Tensor = None,
):
    record = {}
    # only record those that has canberry in chop, or if it is starting state
    for key in env.scene.keys():
        asset = env.scene[key]
        if asset is None:
            continue
        if isinstance(asset, Articulation):
            record[key] = {}
            record[key]["joint_pos"] = asset.data.joint_pos.clone()
            record[key]["joint_vel"] = asset.data.joint_vel.clone()
        elif isinstance(asset, RigidObject):
            record[key] = {}
            record[key]["transform"] = torch.cat([asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w], dim=-1).clone()
            record[key]["velocity"] = asset.data.body_vel_w.clone()
        elif isinstance(asset, Deformable):
            record[key] = {}
            record[key]["nodal_positions"] = asset.data.root_nodal_positions.clone() - env.scene.env_origins
            record[key]["nodal_velocities"] = asset.data.root_nodal_velocities.clone()
        else:
            continue
    return record


def reset_from_state_configuration(
    env: HebiRLTaskEnv,
    env_ids: torch.Tensor,
):
    if not ("trajectory_memory" in env.trajectory_memory.keys()):
        return
    else:
        origins = env.scene.env_origins
        num_memory = len(env.trajectory_memory["trajectory_memory"])
        memory_index = torch.randint(0, num_memory, ())
        memory = env.trajectory_memory["trajectory_memory"][memory_index]
        for key in memory:
            if key in "episode_reward":
                continue
            asset = env.scene[key]
            memkey = memory[key]
            if isinstance(asset, Articulation):
                # joint_pos_ = torch.tensor([-2.6504,  1.1869,  1.6892,  0.4732,  0.4764,  0.4204, -0.5930], device = env.device)
                # asset.write_joint_state_to_sim(joint_pos_, memkey["joint_vel"].clone(), env_ids=env_ids)
                asset.write_joint_state_to_sim(memkey["joint_pos"].clone(), memkey["joint_vel"].clone(), env_ids=env_ids)
                joint_pos = memkey["joint_pos"]
                print(f"written when robot joint pos is {joint_pos}")
            elif isinstance(asset, RigidObject):
                transform = memkey["transform"].clone().repeat(env.num_envs, 1)
                transform[:, :3] += origins
                asset.write_root_pose_to_sim(transform)
                asset.write_root_velocity_to_sim(memkey["velocity"].clone())
            elif isinstance(asset, Deformable):
                asset.root_physx_view.set_sim_nodal_positions(memkey["nodal_positions"].clone() + origins, indices=env_ids)
                asset.root_physx_view.set_sim_nodal_velocities(memkey["nodal_velocities"].clone(), indices=env_ids)
            else:
                continue


def reset_from_demostration(
    env: HebiRLTaskEnv,
    env_ids: torch.Tensor,
):
    demostrations_state = []
    if not ("demostration_memory" in env.trajectory_memory.keys()):
        with h5py.File("logs/robomimic/IkDeltaDls-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/2024-04-26_22-12-55/hdf_dataset.hdf5", "r") as f:
            for demo_key in f["mask/successful_valid"]:
                demo_trajectory = {}
                demo_key_string = demo_key.decode()
                # demo_key_string = 'demo_107'
                observation_data = f[f"data/{demo_key_string}/record"]
                for key in observation_data:
                    demo_trajectory[key] = torch.from_numpy(f[f"data/{demo_key_string}/record/{key}"][:]).to(env.device)
                demostrations_state.append(demo_trajectory)
        env.trajectory_memory["demostration_memory"] = demostrations_state

    else:
        origins = env.scene.env_origins
        num_trajectorys = len(env.trajectory_memory["demostration_memory"])
        trajectorys_index = torch.randint(0, num_trajectorys, ())
        trajectory = env.trajectory_memory["demostration_memory"][trajectorys_index]
        num_states = len(next(iter(trajectory.values())))
        # memory_index = torch.randint(0, num_states,())
        memory_index = torch.randint(int(num_states * 0.2), int(num_states * 0.9), ())
        traj = {}
        for key in trajectory.keys():
            asset_key, prop = key.rsplit('-', 1)
            if not (asset_key in traj.keys()):
                traj[asset_key] = {}
            traj[asset_key][prop] = trajectory[key]

        for key in traj.keys():
            asset = env.scene[key]
            memkey = traj[key]
            if isinstance(asset, Articulation):
                asset.write_joint_state_to_sim(memkey["joint_pos"][memory_index].clone(), memkey["joint_vel"][memory_index].clone(), env_ids=env_ids)
            elif isinstance(asset, RigidObject):
                transform = memkey["transform"][memory_index].clone().repeat(len(env_ids), 1)
                transform[:, :3] += origins[env_ids]
                asset.write_root_pose_to_sim(transform, env_ids=env_ids)
                asset.write_root_velocity_to_sim(memkey["velocity"][memory_index].clone(), env_ids=env_ids)
            elif isinstance(asset, Deformable):
                nodal_pos = memkey["nodal_positions"][memory_index].clone().repeat(len(env_ids), 1)
                nodal_pos[:, :3] += origins[env_ids]
                asset.root_physx_view.set_sim_nodal_positions(nodal_pos, indices=env_ids)
                asset.root_physx_view.set_sim_nodal_velocities(memkey["nodal_velocities"].clone(), indices=env_ids)
            else:
                continue
