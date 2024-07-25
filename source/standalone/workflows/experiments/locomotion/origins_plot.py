# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import subprocess
# local imports
from source.standalone.workflows.rsl_rl import cli_args  # isort: skip
import matplotlib.pyplot as plt
import cv2
import numpy as np
import h5py
import re
import os
import glob
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="IkAbsoluteDls-IdealPDHebi-JointPos-GoalTracking", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True


tasks = [
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Rgh-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Rgh-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Box-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Iprmd-Box-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Box-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Rgh-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-Hfslp-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Prmd-Iprmd-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Iprmd-Box-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Box-Rgh-v0",
    # "Isaac-Velocity-Rough-Anymal-C-Rgh-Hfslp-v0",
    "Isaac-Velocity-Rough-Anymal-C-Flat-v0",
]

translate_dict = {
    "Prmd" : "pyramid_stairs",
    "Iprmd" : "pyramid_stairs_inv",
    "Box" : "boxes",
    "Rgh" : "random_rough",
    "Hfslp" : "hf_pyramid_slope",
    "Flat" : "flat"
}
pattern = re.compile(r"Anymal-C-(.*?)-v0")

filtered_words = [pattern.search(task).group(1) for task in tasks if pattern.search(task)]

env_desc = []
for word in filtered_words:
    env_desc.append("+".join([translate_dict[word] for word in word.split("-")]))


def run_experiment():

    for task, word in zip(tasks, env_desc):
        subprocess.run(
            [
                "python",
                "../IsaacLab/source/standalone/workflows/rsl_rl/train.py",
                "--task",
                task,
                "--headless",
                "--num_envs",
                "4096"
            ]
        )
        logpath = "logs/terrain_experiments/" + word
        # find the h5 file named as env_data_*.h5 as the input
        h5_files = glob.glob(os.path.join(logpath, "env_data_*.h5"))
        if not h5_files:
            print(f"No HDF5 files found in {logpath}")
            continue

        input_h5_file = h5_files[-1]  # Assuming there's only one HDF5 file
        ouput_video = os.path.join(logpath, "video.avi")
        create_video_from_h5(input_h5_file, ouput_video)


def save_frames_from_rgb_h5(h5_filepath, frames_dir="frames2", frame_skip=1):
    with h5py.File(h5_filepath, 'r') as h5_file:
        # Assuming the structure is: h5_file['data'][...]['rgb']
        dataset = h5_file['data']
        
        frame_count = 0
        total_frames = len(dataset)  # Get the total number of frames for the progress bar

        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        for idx, jpeg_dict in enumerate(tqdm(dataset.values(), total=total_frames, desc="Saving frames")):
            if frame_count % frame_skip == 0:
                jpeg = jpeg_dict['rgb'][0]
                jpeg_data_np = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(jpeg_data_np, cv2.IMREAD_COLOR)

                # Save the frame to disk for inspection
                frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(frame_path, frame)

            frame_count += 1

def create_video_from_rgb_h5(h5_filepath, output_video_path, fps=25, frame_skip=1, frame_size=None):
    with h5py.File(h5_filepath, 'r') as h5_file:
        # Assuming the structure is: h5_file['data'][...]['rgb']
        dataset = h5_file['data']
        # Get the first image to determine the frame size if not provided
        if frame_size is None:
            for jpeg_dict in dataset.values():
                jpeg = jpeg_dict['rgb'][0]
                jpeg_data_np = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(jpeg_data_np, cv2.IMREAD_COLOR)
                frame_size = (frame.shape[1], frame.shape[0])
                break
        
        # Prepare the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        frame_count = 0
        total_frames = len(dataset)  # Get the total number of frames for the progress bar

        for jpeg_dict in tqdm(dataset.values(), total=total_frames, desc="Creating video"):
            if frame_count % frame_skip == 0:
                jpeg = jpeg_dict['rgb'][0]
                jpeg_data_np = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(jpeg_data_np, cv2.IMREAD_COLOR)
                video_writer.write(frame)
            frame_count += 1

        video_writer.release()


def create_video_from_h5(h5_filepath, output_video_path, frame_size=(800, 800), fps=25, frame_skip=1):
    # Open the HDF5 file
    metadata = os.path.dirname(h5_filepath).split("/")[-1]
    metadata_list = metadata.split('+')[::-1]
    with h5py.File(h5_filepath, 'r') as h5_file:
        dataset = h5_file['pos_w']
        num_steps = dataset.shape[0]

        # Find global min and max for x and y coordinates
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for step in range(0, num_steps, frame_skip):
            xy_data = dataset[step, :, :2]
            x_min = min(x_min, xy_data[:, 0].min())
            y_min = min(y_min, xy_data[:, 1].min())
            x_max = max(x_max, xy_data[:, 0].max())
            y_max = max(y_max, xy_data[:, 1].max())

        # Prepare the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        print(f"data loaded, {metadata}")
        print("creating videos.....")
        for step in range(0, num_steps, frame_skip):
            xy_data = dataset[step, :, :2]
            x_min = min(x_min, xy_data[:, 0].min())
            y_min = min(y_min, xy_data[:, 1].min())
            x_max = max(x_max, xy_data[:, 0].max())
            y_max = max(y_max, xy_data[:, 1].max())

        # Prepare the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        print("creating videos.....")
        for step in tqdm(range(0, num_steps, frame_skip)):
            # Extract the x and y coordinates
            xy_data = dataset[step, :, :2]

            # Plot the data
            fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size to make space for text
            title = f"Step {step + 1}/{num_steps}"
            ax.scatter(xy_data[:, 0], xy_data[:, 1], s=1)
            ax.set_title(title)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            # Overlay metadata text onto the figure, outside the plot area
            half_length = len(metadata_list) / 2.0
            y0, dy = 0.45 + 0.1 * half_length, 0.1  # Starting y position and vertical space between lines
            for i, line in enumerate(metadata_list):
                fig.text(0.01, y0 - i * dy, line, fontsize=10, color='black', ha='left')

            # Convert the Matplotlib plot to a numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)

            # Convert RGB to BGR format for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, frame_size)  # Resize frame to match video size

            video_writer.write(img)

        # Release the video writer
        video_writer.release()


def main():
    save_frames_from_rgb_h5("logs/rgb/screen_record.hdf5")
    # create_video_from_rgb_h5("logs/rgb/screen_record.hdf5", "logs/rgb/screen_record.avi")
    # run_experiment()


if __name__ == "__main__":
    main()
    # create_video_from_h5(
    #     "logs/terrain_experiments/pyramid_stairs+pyramid_stairs_inv+boxes+random_rough+hf_pyramid_slope/env_data_20240719_214849.h5",
    #     "logs/terrain_experiments/pyramid_stairs+pyramid_stairs_inv+boxes+random_rough+hf_pyramid_slope/video.avi")
