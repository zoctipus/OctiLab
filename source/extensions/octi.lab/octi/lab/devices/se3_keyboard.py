from omni.isaac.lab.devices import Se3Keyboard
import torch
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_mul, matrix_from_quat


class Se3KeyboardDelta(Se3Keyboard):
    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, device="cuda:0"):
        super().__init__(pos_sensitivity, rot_sensitivity)
        self.device = device
        self.delta_pose = torch.zeros(7, device=self.device).unsqueeze(0)

    def reset(self):
        super().reset()

    def advance(self):
        delta_pose, gripper_command = super().advance()

        delta_pose = delta_pose.astype("float32")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=self.device).view(1, -1)

        rot_actions = delta_pose[:, 3:6]
        angle = torch.linalg.vector_norm(rot_actions, dim=1)
        axis = rot_actions / angle.unsqueeze(-1)
        # change from axis-angle to quat convention
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        rot_delta_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6, quat_from_angle_axis(angle, axis), identity_quat
        )

        self.delta_pose[:, :3] = delta_pose[:, :3]
        self.delta_pose[:, 3:7] = rot_delta_quat

        return self.delta_pose, gripper_command


class Se3KeyboardAbsolute(Se3Keyboard):
    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, device="cuda:0"):
        super().__init__(pos_sensitivity, rot_sensitivity)
        self.device = device
        self.absolute_pose = torch.zeros(7, device=self.device)
        # self.init_pose = torch.tensor([[-0.265, -0.28, 0.05, 0.0, 0.0, 1, 0.0]], device=self.device)
        self.init_pose = torch.tensor([[-0.3000, -0.3000, 0.0500, -1.5497e-05, 2.0993e-05, 7.1197e-01, -7.0221e-01]], device = self.device)

    def reset(self):
        super().reset()
        self.absolute_pose = self.init_pose.clone()

    def advance(self):
        delta_pose, gripper_command = super().advance()

        delta_pose = delta_pose.astype("float32")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=self.device).view(1, -1)

        rot_actions = delta_pose[:, 3:6]
        angle = torch.linalg.vector_norm(rot_actions, dim=1)
        axis = rot_actions / angle.unsqueeze(-1)
        # change from axis-angle to quat convention
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        rot_delta_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6, quat_from_angle_axis(angle, axis), identity_quat
        )
        new_position = apply_local_translation(self.absolute_pose[:, :3], delta_pose[:, :3], self.absolute_pose[:, 3:])
        self.absolute_pose[:, :3] = new_position
        self.absolute_pose[:, 3:7] = quat_mul(rot_delta_quat, self.absolute_pose[:, 3:7])

        return self.absolute_pose, gripper_command


def apply_local_translation(current_position, local_translation, orientation_quaternion):
    # Assuming matrix_from_quat correctly handles batch inputs and outputs a batch of rotation matrices
    rotation_matrix = matrix_from_quat(orientation_quaternion)  # Expected shape (n, 3, 3)

    # Ensure local_translation is correctly shaped for batch matrix multiplication
    local_translation = local_translation.unsqueeze(-1)  # Shape becomes (n, 3, 1) for matmul

    local_translation[:, [1, 2]] = -local_translation[:, [2, 1]]
    # Rotate the local translation vector to align with the global frame
    global_translation = torch.matmul(rotation_matrix, local_translation).squeeze(-1)  # Back to shape (n, 3)

    # Apply the translated vector to the object's current position
    new_position = current_position + global_translation

    return new_position
