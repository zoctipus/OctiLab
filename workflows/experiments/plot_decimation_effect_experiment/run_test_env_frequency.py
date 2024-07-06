# run_simulations.py
import subprocess
import numpy as np
import matplotlib.pyplot as plt

decimations = [2, 5, 10]


def run_experiment():

    for decimation in decimations:
        subprocess.run(
            [
                "python",
                "workflows/tests/test_env_frequency.py",
                "--task",
                "IkAbsoluteDls-IdealPDHebi-JointPos-GoalTracking-v0",
                "--decimation",
                str(decimation),
                "--headless",
            ]
        )


def plot_joint_target_pos():
    plt.figure(figsize=(15, 20))  # Create a large figure to accommodate all subplots

    line_styles = ['-', '--', '-.', ':']  # Different line styles for variety

    for joint in range(7):  # Assuming there are 7 degrees of freedom
        plt.subplot(7, 1, joint + 1)  # Create a subplot for each joint
        for idx, decimation in enumerate(decimations):
            joint_target_positions = np.load(f"logs/test/joint_pos_target_{decimation}.npy")
            joint_positions = np.load(f"logs/test/current_pos_target_{decimation}.npy")

            color = f"C{idx}"  # Matplotlib default color cycle

            # plt.plot(joint_target_positions[:, 0, joint], label=f"Target Position,Decimation={decimation}", linestyle='-', color=color)
            plt.plot(joint_positions[:, 0, joint], label=f"Decimation={decimation}", linestyle='--', color=color)

        plt.xlabel("Agent Steps(50hz)")
        plt.ylabel(f"Joint {joint + 1} Position, Radian")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_position():
    # Now load the results and plot them
    plt.figure(figsize=(15, 10))

    for decimation in decimations:
        positions = np.load(f"logs/test/positions_decimation_{decimation}.npy")
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        z_positions = positions[:, 2]

        plt.subplot(3, 1, 1)
        plt.plot(x_positions, label=f"X Position (decimation={decimation})")

        plt.subplot(3, 1, 2)
        plt.plot(y_positions, label=f"Y Position (decimation={decimation})")

        plt.subplot(3, 1, 3)
        plt.plot(z_positions, label=f"Z Position (decimation={decimation})")

    plt.subplot(3, 1, 1)
    plt.xlabel("Index")
    plt.ylabel("X Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.xlabel("Index")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.xlabel("Index")
    plt.ylabel("Z Position")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def compare():
    data = []
    for decimation in decimations:
        positions = np.load(f"logs/test/positions_decimation_{decimation}.npy")
        data.append(positions)
    print(data)


if __name__ == "__main__":
    run_experiment()
    # plot_joint_target_pos()
    # plot_position()
