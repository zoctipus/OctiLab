from __future__ import annotations

import argparse
import h5py
import numpy as np

def copy_and_reshape_data(src_file, dest_file):
    with h5py.File(src_file, 'r') as src, h5py.File(dest_file, 'w') as dest:
        # Function to recursively copy and optionally reshape data
        def recursive_copy(src_grp, dest_grp):
            for k, v in src_grp.items():
                if isinstance(v, h5py.Dataset):
                    # Check if this dataset needs reshaping (and reshape if necessary)
                    if k in ['base_picture', 'wrist_picture']:
                        # Example: Reshape only specific datasets
                        reshaped_data = np.array(v).reshape(-1, 120, 160, 3)
                        dest_grp.create_dataset(k, data=reshaped_data)
                    else:
                        # Copy other datasets as is
                        src_grp.copy(k, dest_grp)
                elif isinstance(v, h5py.Group):
                    # Recreate the group structure in the destination file
                    dest_subgrp = dest_grp.create_group(k)
                    # Recursively copy/reshape for this subgroup
                    recursive_copy(v, dest_subgrp)

        # Start the recursive copy/reshape process from the root of the HDF5 file
        recursive_copy(src, dest)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Copy and optionally reshape specific datasets in an HDF5 file.")
    parser.add_argument("src_file", type=str, help="The path to the source HDF5 file.")
    parser.add_argument("dest_file", type=str, help="The path to the destination HDF5 file.")
    args = parser.parse_args()

    # Perform the copy and reshape operation
    copy_and_reshape_data(args.src_file, args.dest_file)
