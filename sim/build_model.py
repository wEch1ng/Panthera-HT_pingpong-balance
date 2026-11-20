"""Build Panthera balance MJCF from the bundled follower URDF."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from sim import urdf_converter
from sim.robot import SCENE, ARM_JOINTS, HOME_Q, PADDLE_INSERTION_M, GRIPPER_GAP_M, GRIPPER_CENTER_Y_M


def build():
    urdf_converter.build()
    tree = ET.parse(SCENE.parent / 'scene.xml')
    root = tree.getroot()
    root.set('model', 'Panthera_HT_balance')
    root.remove(root.find('keyframe'))
    root.remove(root.find('actuator'))
    root.find('option').set('timestep', str(1 / 480))
    root.find('option').set('cone', 'elliptic')
    asset, world = root.find('asset'), root.find('worldbody')
    original = ET.parse(SCENE.parent.parent / 'paddle/paddle.xml').getroot()
    for item in original.find('asset'):
        if item.get('name', '').startswith('paddle_'):
            node = deepcopy(item)
            if node.tag == 'mesh':
                node.set('file', '../paddle/' + node.get('file'))
            asset.append(node)
    # The inherited paddle tool frame has -Z forward and +Y face normal.
    # Panthera uses +X forward and +Z up. The handle clamp is 68 mm from
    # this reference, aligning its clamp point with gripper_center_link.
    # Close the fixed fingers symmetrically onto the measured 30 mm handle.
    # The URDF itself keeps its original open pose for other applications.
    for name, sign in [("gripper_L_link", 1), ("gripper_R_link", -1)]:
        finger = world.find(f".//body[@name='{name}']")
        pos = np.fromstring(finger.get('pos'), sep=' ')
        pos[1] = GRIPPER_CENTER_Y_M + sign * GRIPPER_GAP_M / 2
        finger.set('pos', ' '.join(map(str, pos)))
    tool = ET.SubElement(world.find(".//body[@name='gripper_center_link']"),
                         'body', name='gripper', pos=f'{-0.068 - PADDLE_INSERTION_M} 0 0', quat='-0.5 -0.5 0.5 0.5')
    tool.append(deepcopy(original.find(".//body[@name='paddle']")))
    camera_body = world.find(".//body[@name='camera_link']")
    if camera_body is None:
        raise ValueError("Enable camera_link and camera_joint in the follower URDF first")
    # CAD camera_link: +Z forward, +X image right, +Y image down.
    # MuJoCo looks along -Z with +Y up. Optical centre/FOV are simulation
    # assumptions until measured intrinsics and optical extrinsics are supplied.
    ET.SubElement(camera_body, 'camera', name='wrist', pos='0 0 0',
                  quat='0 1 0 0', fovy='58', mode='fixed')
    for g in tool.iter('geom'):
        g.attrib.pop('class', None)
        g.set('contype', '4' if g.get('name') == 'paddle_blade' else '0')
        g.set('conaffinity', '8' if g.get('name') == 'paddle_blade' else '0')
        if g.get('type') == 'mesh':
            g.set('group', '1'); g.set('density', '0')
    ball = deepcopy(original.find(".//body[@name='ball']"))
    ball.find('geom').set('contype', '8')
    ball.find('geom').set('conaffinity', '5')
    world.append(ball)
    world.find("geom[@name='floor']").set('conaffinity', '10')
    actuator = ET.SubElement(root, 'actuator')
    for name, kp, kv in zip(ARM_JOINTS, [240, 360, 300, 180, 100, 80], [18, 24, 20, 12, 8, 6]):
        joint = world.find(f".//joint[@name='{name}']")
        ET.SubElement(actuator, 'position', name=name, joint=name, kp=str(kp), kv=str(kv),
                      ctrlrange=joint.get('range'), forcerange=joint.get('actuatorfrcrange'))
    ET.indent(tree, space='  ')
    tree.write(SCENE, encoding='utf-8', xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    d.qpos[:6] = HOME_Q
    mujoco.mj_forward(m, d)
    normal = d.geom('paddle_blade').xmat.reshape(3, 3)[:, 2]
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-5)
    print(f'Built {SCENE}: {m.nu} actuators, blade centre {d.geom("paddle_blade").xpos}, normal {normal}')
    return m


if __name__ == '__main__':
    build()
