# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import torch
import h5py
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def main():
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path="logs/robomimic/CustomIkAbsolute-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/bc/20240410110049/models/model_epoch_100.pth", 
        device=device, verbose=True)
    count = 0
    observation = {}
    true_action = []
    with h5py.File("logs/robomimic/CustomIkAbsolute-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/hdf_dataset.hdf5", "r") as f:
        observation_data = f["data/demo_0/obs"]
        for key in observation_data:
            observation[key] = torch.from_numpy(f[f"data/demo_0/obs/{key}"][:]).to(device)
        true_action = torch.from_numpy(f[f"data/demo_0/actions"][:]).to(device)
    error_sum = torch.zeros(true_action.shape[1], device = device)
    error_percentage_sum = torch.zeros(true_action.shape[1], device = device)
    error_percentage_average = torch.zeros(true_action.shape[1], device = device)
        # run everything in inference mode
    for _ in range(len(true_action)):
        with torch.inference_mode():
            hacked_obs = {}
            for key, value in observation.items():
                hacked_obs[key] = value[count].view(1, -1)
            # compute actions
            predicted_actions = policy(hacked_obs)
            predicted_actions = torch.from_numpy(predicted_actions).to(device=device).view(1, -1)
            error = torch.abs(predicted_actions - true_action[count].view(1, -1))
            error_percentage = (error / (true_action[count].view(1, -1))).clamp(max=1)
            error_sum += error[0]
            error_percentage_sum += error_percentage[0]
            error_percentage_average = error_percentage_sum/count
            count += 1
            print(error_percentage_average)


if __name__ == "__main__":
    # run the main function
    main()
