import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
# Example points in a NumPy array
# Replace this with your actual array of points

trajectories = []
with h5py.File("logs/robomimic/CustomIkAbsolute-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/2024-04-10_13-11-30/hdf_dataset.hdf5", "r") as f:
    for demo_key in f[f"mask/successful_valid"]:
        demo_key_string = demo_key.decode()
        # Extract the action trajectory
        action_traj = f[f"data/{demo_key_string}/actions"][:, :3]

        trajectories.append(action_traj)

# Now, plot each trajectory with its own color
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory in the list
count = 0
for traj in trajectories:
    count += 1
    if count > 100:
        break
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], s=1)  # s=1 for small point size

# Adding labels to the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Display the plot
plt.show()