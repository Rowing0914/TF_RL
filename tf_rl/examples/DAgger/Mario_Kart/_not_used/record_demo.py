"""
==== joystick input ====
x        = self.JoystickX
y        = self.JoystickY
forward  = self.A
jump     = self.RightBumper
use_item = self.C_left
"""

from inputs import get_gamepad
import mss, time, math, threading


def _monitor_controller():
    while True:
        events = get_gamepad()
        for event in events:
            print(event.code, event.state)


if __name__ == '__main__':
    _monitor_controller()