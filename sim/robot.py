"""Panthera follower configuration; radians, metres, newton metres."""
from pathlib import Path
import numpy as np

ROBOT_ID = "panthera-ht-v1.1.0"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
N_ARM = len(ARM_JOINTS)
BASE_OBS = 4 + 2 * N_ARM
HOME_Q = np.array([0., 1.0, 1.7, -0.7, 0., 0.])
SCENE = Path(__file__).resolve().parents[1] / "assets/panthera/scene_balance.xml"
DESCRIPTION = Path(__file__).resolve().parents[1] / 'assets/description'
SOURCE_URDF = DESCRIPTION / 'urdf/Panthera-HT_description_V1.1.0_follower.urdf'

CAMERA_ID = "d405-urdf-optical-v1"

# Rigid paddle mount: insert 30 mm farther into the fixed fingers.
PADDLE_INSERTION_M = 0.030
GRIPPER_GAP_M = 0.030
GRIPPER_CENTER_Y_M = -0.000211
