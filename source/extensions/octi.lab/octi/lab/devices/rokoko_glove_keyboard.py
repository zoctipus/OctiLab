from omni.isaac.lab.devices import DeviceBase
from collections.abc import Callable
from . import RokokoGlove
from . import Se3KeyboardAbsolute


class RokokoGloveKeyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as absolute poses, and gloves for sending commands
    of hands individual part.

    The keyboard command binding:
    * absolute pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    """
    def __init__(self,
                 pos_sensitivity: float = 0.4,
                 rot_sensitivity: float = 0.8,
                 UDP_IP: str = "0.0.0.0",  # Listen on all available network interfaces
                 UDP_PORT: int = 14043,   # Make sure this matches the port used in Rokoko Studio Live
                 left_hand_track: list[str] = [],
                 right_hand_track: list[str] = [],
                 scale: float = 1,
                 proximal_offset: float = 0.3,
                 init_pose: list[float] = [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 device="cuda:0"):
        """Initialize the Rokoko_Glove Controller and Keyboard Controller.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
            UDP_IP: The IP Address of network to listen to, 0.0.0.0 refers to all available networks
            UDP_PORT: The port Rokoko Studio Live sends to
            left_hand_track: the trackpoint of left hand this class will be tracking.
            right_hand_track: the trackpoint of right hand this class will be tracking.
            scale: the overal scale for the hand.
            proximal_offset: the inter proximal offset that shorten or widen the spread of hand.
            init_pose: the initial pose command when environment resets.
        """
        self.hand_device = RokokoGlove(UDP_IP, UDP_PORT, left_hand_track, right_hand_track, scale, proximal_offset, device)
        self.keyboard_device = Se3KeyboardAbsolute(pos_sensitivity, rot_sensitivity, init_pose, device)

    def __del__(self):
        self.keyboard_device.__del__()

    def __str__(self):
        return self.keyboard_device.__str__()

    def reset(self):
        """Resets the internal states of hand and keyboard"""
        self.hand_device.reset()
        self.keyboard_device.reset()

    def advance(self):
        """Gloves no longer translate hands, intead, keyboard controls hands translation
        While Gloves providing the properly scaled, ordered, selected tracking results received from Rokoko Studio,
        gloves does not translates hands, instead keyboard provides palms position.

        Returns:
            A tuple containing the 2D (n,7) pose array ordered by user inputed joint track list, and a dummy truth value.
        """
        hand_command = self.hand_device.advance()[0]
        keyboard_command = self.keyboard_device.advance()[0]
        hand_command[:, :3] -= hand_command[0, :3]
        hand_command[:, :3] += keyboard_command[0, [0, 2, 1]]
        return hand_command, True

    def debug_advance_all_joint_data(self):
        hand_command = self.hand_device.debug_advance_all_joint_data()[0]
        keyboard_command = self.keyboard_device.advance()[0]
        hand_command[:, :3] -= hand_command[21, :3]
        hand_command[:, :3] += keyboard_command[0, [0, 2, 1]]
        return hand_command, True


    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        self.keyboard_device.add_callback(key, func)
