# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Script for splitting a dataset hdf5 file into training and validation trajectories.

Args:
    dataset: path to hdf5 dataset

    filter_key: if provided, split the subset of trajectories
        in the file that correspond to this filter key into a training
        and validation set of trajectories, instead of splitting the
        full set of trajectories

    ratio: validation ratio, in (0, 1). Defaults to 0.1, which is 10%.

Example usage:
    python split_train_val.py --dataset /path/to/demo.hdf5 --ratio 0.1
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np


def filter_success(hdf5_path: str, max_length = -1, filter_key=None, remove_actionless=False):
    """
    Splits data into training set and validation set from HDF5 file.

    Args:
        hdf5_path: path to the hdf5 file to load the transitions from
        val_ratio: ratio of validation demonstrations to all demonstrations

        filter_key: if provided, split the subset of demonstration keys stored
            under mask/@filter_key instead of the full set of demonstrations
    """
    # retrieve number of demos
    
    shape_consistency = {
        'statue': -1,
        'obs' : -1,
        'next_obs' : -1,
        'success' : -1,
        'dones' : -1,
        'rewards' : -1,
        'actions' : -1,
    }
    f = h5py.File(hdf5_path, "r+")
    success_mask = []
    count = 0
    success_sample_count = 0
    failure_sample_count = 0
    for demo_key in f['data'].keys():
        success = f['data'][demo_key]['success'][(-1)]
        if success:
            actions = f['data'][demo_key]["actions"][:]
            successes = f['data'][demo_key]["success"][:]
            valid_steps = (np.sum(actions[:, :3], axis=1) > 0.001) | (successes == 1)
            # Modify datasets only if the demo is successful
            if any(valid_steps):
                # Delete existing datasets
                if remove_actionless:
                    obs_tmp =  f['data'][demo_key]['obs'][valid_steps]
                    next_obs_tmp =  f['data'][demo_key]['next_obs'][valid_steps]
                    success_tmp =  f['data'][demo_key]['success'][valid_steps]
                    dones_tmp =  f['data'][demo_key]['dones'][valid_steps]
                    rewards_tmp =  f['data'][demo_key]['rewards'][valid_steps]
                    actions_tmp =  f['data'][demo_key]['actions'][valid_steps]
                else:
                    obs_tmp =  f['data'][demo_key]['obs'][:]
                    next_obs_tmp =  f['data'][demo_key]['next_obs'][:]
                    success_tmp =  f['data'][demo_key]['success'][:]
                    dones_tmp =  f['data'][demo_key]['dones'][:]
                    rewards_tmp =  f['data'][demo_key]['rewards'][:]
                    actions_tmp =  f['data'][demo_key]['actions'][:]
                

                             
                # Delete existing datasets if they exist
                dataset_paths = [
                    'obs', 'next_obs', 'success', 'dones', 'rewards', 'actions'
                ]
                for dataset_path in dataset_paths:
                    full_path = f'data/{demo_key}/{dataset_path}'
                    if dataset_path in f['data'][demo_key]:
                        del f[full_path]

                # Ensure groups for nested data structures exist
                for subkey in ["obs", "next_obs"]:
                    if subkey not in f['data'][demo_key]:
                        f['data'][demo_key].create_group(subkey)

                # Recreate datasets within their specific groups
                f['data'][demo_key]['obs'].create_dataset("state", data=obs_tmp)
                f['data'][demo_key]['next_obs'].create_dataset("state", data=next_obs_tmp)
                f['data'][demo_key].create_dataset("success", data=success_tmp)
                f['data'][demo_key].create_dataset("dones", data=dones_tmp)
                f['data'][demo_key].create_dataset("rewards", data=rewards_tmp)
                f['data'][demo_key].create_dataset("actions", data=actions_tmp)

            # Check if trajectory length is within the specified max length
            success_step_samples = len(f['data'][demo_key]['success'])
            if f['data'][demo_key]['success'][-1] and max_length != -1 and success_step_samples < max_length:
                success_mask.append(demo_key.encode('utf-8'))
                print(f"{demo_key} has {success_step_samples} steps is stored as successful")
                success_sample_count += success_step_samples
            else:
                print(f"{demo_key} has {success_step_samples} steps is not stored as successful")
                failure_sample_count += success_step_samples
            f['data'][demo_key].attrs["num_samples"] = success_step_samples
            
    
    # Check if the "mask" group exists and delete it if it does
    if "mask" in f:
        del f["mask"]

    # Create a new "mask" group and add the "successful" dataset
    mask_group = f.create_group("mask")
    print(f"total success samples: {success_sample_count}")
    print(f"total failure samples: {failure_sample_count}")
    mask_group.create_dataset("successful", data=np.array(success_mask))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("--remove_actionless", action='store_true', help="whether to remove actionless state action pair")
    parser.add_argument("--max_episode_length", type=int, default=-1, help="the maximum length allowed for successful episode")
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help=(
            "If provided, split the subset of trajectories in the file that correspond to this filter key"
            " into a training and validation set of trajectories, instead of splitting the full set of"
            " trajectories."
        ),
    )
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    filter_success(args.dataset, max_length=args.max_episode_length, filter_key=args.filter_key, remove_actionless=args.remove_actionless)
