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
    count = 0
    observation = {}
    with h5py.File("logs/robomimic/CustomIkAbsolute-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/2024-04-13_17-06-42/hdf_dataset.hdf5", "r") as f:
        for demo_key in f["mask/successful"]:
            demo_key_string = demo_key.decode()
            observation_data = f[f"data/{demo_key_string}/obs"]
            # observation_data = f[f"data/{demo_key[0]}/obs"]
            for key in observation_data:
                observation[key] = torch.from_numpy(f[f"data/{demo_key_string}/obs/{key}"][:]).to(device)
            action = torch.from_numpy(f[f"data/{demo_key_string}/actions"][:]).to(device)
            
            max_act = torch.max(action[:, :3])
            min_act = torch.min(action[:, :3])

if __name__ == "__main__":
    # run the main function
    main()
