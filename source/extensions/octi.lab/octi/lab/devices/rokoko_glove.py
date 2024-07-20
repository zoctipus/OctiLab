import socket
import lz4.frame
from omni.isaac.lab.devices import DeviceBase
import json
import torch
from collections.abc import Callable


class RokokoGlove(DeviceBase):
    """A Rokoko_Glove controller for sending SE(3) commands as absolute poses of hands individual part
    This class is designed to track hands and fingers's pose from rokoko gloves.
    It uses the udp network protocal to listen to Rokoko Live Studio data gathered from Rokoko smart gloves,
    and process the data in form of torch Tensor.
    Addressing the efficiency and ease to understand, the tracking will only be performed with user's parts
    input, and all Literal of available parts is exhaustively listed in the comment under method __init__.

    available tracking literals:
        LEFT_HAND:
            leftHand, leftThumbProximal, leftThumbMedial, leftThumbDistal, leftThumbTip,
            leftIndexProximal, leftIndexMedial, leftIndexDistal, leftIndexTip,
            leftMiddleProximal, leftMiddleMedial, leftMiddleDistal, leftMiddleTip,
            leftRingProximal, leftRingMedial, leftRingDistal, leftRingTip,
            leftLittleProximal, leftLittleMedial, leftLittleDistal, leftLittleTip

        RIGHT_HAND:
            rightHand, rightThumbProximal, rightThumbMedial, rightThumbDistal, rightThumbTip
            rightIndexProximal, rightIndexMedial, rightIndexDistal, rightIndexTip,
            rightMiddleProximal, rightMiddleMedial, rightMiddleDistal, rightMiddleTip
            rightRingProximal, rightRingMedial, rightRingDistal, rightRingTip,
            rightLittleProximal, rightLittleMedial, rightLittleDistal, rightLittleTip
    """
    def __init__(self,
                 UDP_IP: str = "0.0.0.0",  # Listen on all available network interfaces
                 UDP_PORT: int = 14043,   # Make sure this matches the port used in Rokoko Studio Live
                 left_hand_track: list[str] = [],
                 right_hand_track: list[str] = [],
                 scale: float = 1.65,
                 proximal_offset: float = 0.5,
                 thumb_scale: float = 1.1,
                 device="cuda:0"):
        """Initialize the Rokoko_Glove Controller.
        Be aware that current implementation outputs pose of each hand part in the same order as input list,
        but parts come from left hand always come before parts from right hand.

        Args:
            UDP_IP: The IP Address of network to listen to, 0.0.0.0 refers to all available networks
            UDP_PORT: The port Rokoko Studio Live sends to
            left_hand_track: the trackpoint of left hand this class will be tracking.
            right_hand_track: the trackpoint of right hand this class will be tracking.
            scale: the overal scale for the hand.
            proximal_offset: the inter proximal offset that shorten or widen the spread of hand.
        """
        self.device = device
        self._additional_callbacks = dict()
        # Define the IP address and port to listen on
        self.UDP_IP = UDP_IP
        self.UDP_PORT = UDP_PORT
        self.scale = scale
        self.proximal_offset = proximal_offset
        self.thumb_scale = thumb_scale
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        self.sock.bind((UDP_IP, UDP_PORT))

        print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

        self.left_hand_joint_names = [
            'leftHand', 'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
            'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
            'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
            'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
            'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip']

        self.right_hand_joint_names = [
            'rightHand', 'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
            'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
            'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
            'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
            'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip']

        self.left_joint_dict = {self.left_hand_joint_names[i] : i for i in range(len(self.left_hand_joint_names))}
        self.right_joint_dict = {self.right_hand_joint_names[i] : i for i in range(len(self.right_hand_joint_names))}

        self.left_fingertip_names = left_hand_track
        self.right_fingertip_names = right_hand_track

        self.left_finger_dict = {i : self.left_joint_dict[i] for i in self.left_fingertip_names}
        self.right_finger_dict = {i : self.right_joint_dict[i] for i in self.right_fingertip_names}

        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)
        self.fingertip_poses = torch.zeros((len(self.left_hand_joint_names) + len(self.right_hand_joint_names), 7), device=self.device)
        output_indices_list = [*[self.right_joint_dict[i] for i in self.left_fingertip_names], *[self.right_joint_dict[i] + len(self.left_hand_joint_names) for i in self.right_fingertip_names]]
        self.output_indices = torch.tensor(output_indices_list, device=self.device)

    def reset(self):
        "Reset Internal Buffer"
        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)

    def advance(self):
        """Provides the properly scaled, ordered, selected tracking results received from Rokoko Studio.

        Returns:
            A tuple containing the 2D (n,7) pose array ordered by user inputed joint track list, and a dummy truth value.
        """
        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)
        body_data = self._get_gloves_data()

        for joint_name in self.left_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.left_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.left_fingertip_poses[self.left_joint_dict[joint_name]][3:] = joint_rotation

        for joint_name in self.right_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][3:] = joint_rotation

        left_wrist_position = self.left_fingertip_poses[0][:3]
        if len(self.left_fingertip_names) > 0:
            # scale
            self.left_fingertip_poses[:, :3] = (self.left_fingertip_poses[:, :3] - left_wrist_position) * self.scale + left_wrist_position
            # reposition
            leftIndexProximalIdx = self.left_joint_dict["leftIndexProximal"]
            leftMiddleProximalIdx = self.left_joint_dict["leftMiddleProximal"]
            leftRingProximalIdx = self.left_joint_dict["leftRingProximal"]
            leftLittleProximalIdx = self.left_joint_dict["leftLittleProximal"]

            reposition_vector = self.left_fingertip_poses[leftMiddleProximalIdx][:3] - self.left_fingertip_poses[leftIndexProximalIdx][:3]
            self.left_fingertip_poses[leftIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.fingertip_poses[:len(self.left_fingertip_poses)] = self.left_fingertip_poses

        right_wrist_position = self.right_fingertip_poses[0][:3]
        if len(self.right_fingertip_names) > 0:
            rightThumbProximalIdx = self.right_joint_dict["rightThumbProximal"]
            rightIndexProximalIdx = self.right_joint_dict["rightIndexProximal"]
            rightMiddleProximalIdx = self.right_joint_dict["rightMiddleProximal"]
            rightRingProximalIdx = self.right_joint_dict["rightRingProximal"]
            rightLittleProximalIdx = self.right_joint_dict["rightLittleProximal"]
            # scale
            self.right_fingertip_poses[:, :3] = (self.right_fingertip_poses[:, :3] - right_wrist_position) * self.scale + right_wrist_position
            t_idx = rightThumbProximalIdx
            self.right_fingertip_poses[t_idx:t_idx + 4, :3] = (self.right_fingertip_poses[t_idx:t_idx + 4, :3] - right_wrist_position) * self.thumb_scale + right_wrist_position
            # reposition
            reposition_vector = self.right_fingertip_poses[rightMiddleProximalIdx][:3] - self.right_fingertip_poses[rightIndexProximalIdx][:3]
            self.right_fingertip_poses[rightIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.fingertip_poses[len(self.left_fingertip_poses):] = self.right_fingertip_poses

        return self.fingertip_poses[self.output_indices], True  # True being a placeholder statisfy abstract method

    def _get_gloves_data(self):
        data, addr = self.sock.recvfrom(8192)  # Buffer size is 1024 bytes
        decompressed_data = lz4.frame.decompress(data)
        received_json = json.loads(decompressed_data)
        body_data = received_json["scene"]["actors"][0]["body"]
        return body_data

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def debug_advance_all_joint_data(self):
        """Provides the properly scaled, all tracking results received from Rokoko Studio.
        It is intended to use a debug and visualization function inspecting all data from Rokoko Glove.

        Returns:
            A tuple containing the 2D (42,7) pose array(left:0-21, right:21-42), and a dummy truth value.
        """
        body_data = self._get_gloves_data()

        # for joint_name in self.left_hand_joint_names:
        #     joint_data = body_data[joint_name]
        #     joint_position = torch.tensor(list(joint_data["position"].values()))
        #     joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
        #     self.left_fingertip_poses[self.left_joint_dict[joint_name]][:3] = joint_position
        #     self.left_fingertip_poses[self.left_joint_dict[joint_name]][3:] = joint_rotation

        for joint_name in self.right_hand_joint_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][3:] = joint_rotation

        # left_wrist_position = self.left_fingertip_poses[0][:3]

        # self.left_fingertip_poses[:, :3] = (self.left_fingertip_poses[:, :3] - left_wrist_position) * self.scale + left_wrist_position
        # self.fingertip_poses[:len(self.left_fingertip_poses)] = self.left_fingertip_poses

        right_wrist_position = self.right_fingertip_poses[0][:3]
        # scale
        rightThumbProximalIdx = self.right_joint_dict["rightThumbProximal"]
        rightIndexProximalIdx = self.right_joint_dict["rightIndexProximal"]
        rightMiddleProximalIdx = self.right_joint_dict["rightMiddleProximal"]
        rightRingProximalIdx = self.right_joint_dict["rightRingProximal"]
        rightLittleProximalIdx = self.right_joint_dict["rightLittleProximal"]
        # scale
        self.right_fingertip_poses[:, :3] = (self.right_fingertip_poses[:, :3] - right_wrist_position) * self.scale + right_wrist_position
        t_idx = rightThumbProximalIdx
        self.right_fingertip_poses[t_idx:t_idx + 4, :3] = (self.right_fingertip_poses[t_idx:t_idx + 4, :3] - right_wrist_position) * self.thumb_scale + right_wrist_position
        # reposition
        reposition_vector = self.right_fingertip_poses[rightMiddleProximalIdx][:3] - self.right_fingertip_poses[rightIndexProximalIdx][:3]
        self.right_fingertip_poses[rightIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.fingertip_poses[len(self.left_fingertip_poses):] = self.right_fingertip_poses

        return self.fingertip_poses, True
