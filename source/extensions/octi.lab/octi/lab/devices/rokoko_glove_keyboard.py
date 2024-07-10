from omni.isaac.lab.devices import DeviceBase
from collections.abc import Callable
from . import RokokoGlove
from . import Se3KeyboardAbsolute


class RokokoGloveKeyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as absolute poses, and gloves for sending commands
    of hands individual part.
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
                 device="cuda:0"):
        self.hand_device = RokokoGlove(UDP_IP, UDP_PORT, left_hand_track, right_hand_track, scale, proximal_offset, device)
        self.keyboard_device = Se3KeyboardAbsolute(pos_sensitivity, rot_sensitivity, device)

    def __del__(self):
        self.keyboard_device.__del__()

    def __str__(self):
        return self.keyboard_device.__str__()

    def reset(self):
        self.hand_device.reset()
        self.keyboard_device.reset()

    def advance(self):
        hand_command = self.hand_device.advance()[0]
        keyboard_command = self.keyboard_device.advance()[0]
        hand_command[:, :3] -= hand_command[0, :3]
        hand_command[:, :3] += keyboard_command[0, [0, 2, 1]]
        return hand_command, True

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        self.keyboard_device.add_callback(key, func)
