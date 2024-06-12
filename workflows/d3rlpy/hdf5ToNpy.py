# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import numpy as np
import h5py


def hdf5_to_npy(hdf5_path, save_npy_path, obs_group_sequence=None):
    
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in f[f"mask/successful"]:
            demo_dict = {
                    'obs': [],
                    'act': [],
                    'rew': [],
                    'done': []
                }
            demo_key_string = demo_key.decode()
            
            if obs_group_sequence is not None:
                obs = []
                for group in obs_group_sequence:
                    obs.append(f[f"data/{demo_key_string}/obs/{group}"][:])
                obs = np.concatenate(obs, axis = -1)
            else:
                obs = f[f"data/{demo_key_string}/obs"][:]
            actions = f[f"data/{demo_key_string}/actions"][:]
            rewards = f[f"data/{demo_key_string}/rewards"][:]
            dones = f[f"data/{demo_key_string}/dones"][:]

            for i in range(len(obs)):
                demo_dict['obs'].append(obs[i])
                demo_dict['act'].append(actions[i])
                demo_dict['rew'].append(rewards[i])
                demo_dict['done'].append(dones[i])
            demos.append(demo_dict)
    np.save(save_npy_path, demos, allow_pickle=True)

