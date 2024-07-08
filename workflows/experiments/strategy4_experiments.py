# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
import subprocess
# local imports
from workflows.rsl_rl import cli_args  # isort: skip

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

tasks = ["IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp0dot5-ep1",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp1dot5-ep1",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp2-ep1",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp5-ep1",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp1-ep0dot2",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp1-ep0dot5",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp1-ep1dot5",
         "IkDeltaDls-Strategy4MotorHebi-JointPos-GoalTracking-pp1-ep2",]


def run_experiment():

    for task in tasks:
        subprocess.run(
            [
                "python",
                "workflows/rsl_rl/train.py",
                "--task",
                task,
                "--headless",
                "--num_envs",
                "4096"
            ]
        )


def main():
    run_experiment()


if __name__ == "__main__":
    main()
