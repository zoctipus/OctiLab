# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool to check structure of hdf5 files."""

from __future__ import annotations

import argparse
import h5py
import os
from PIL import Image


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Check structure of hdf5 file.")
    parser.add_argument("file", type=str, default=None, help="The path to HDF5 file to analyze.")
    args_cli = parser.parse_args()
    include_alpha_channel = False
    # open specified file
    with h5py.File(args_cli.file, "r") as f:
        # print name of the file first
        data_flat = f["data/demo_0/obs/base_picture"][:]
        data_images = data_flat.reshape(-1, 120, 160, 3)

    save_dir = os.path.dirname(args_cli.file)
    save_dir = os.path.join(save_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(data_images):
        if include_alpha_channel:
            # Save the image with the alpha channel (RGBA)
            img = Image.fromarray(image, 'RGBA')
        else:
            # Convert RGBA to RGB (discard the alpha channel) and save
            image_rgb = image[:, :, :3]  # Discard the alpha channel
            img = Image.fromarray(image_rgb, 'RGB')
        
        # Save the image
        img.save(os.path.join(save_dir, f"image_{i}{'_rgba' if include_alpha_channel else '_rgb'}.png"))
