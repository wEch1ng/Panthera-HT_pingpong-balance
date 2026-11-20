"""URDF/mesh conversion adapted from the supplied Panthera description viewer."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from sim.robot import DESCRIPTION as ROOT, SOURCE_URDF as SOURCE
OUTPUT = Path(__file__).resolve().parents[1] / "assets/panthera"
OBJ_DIR = OUTPUT / "meshes" / "obj"
HOME = np.zeros(6)


def build():
    """Convert the source follower URDF and GLB/STL meshes into local MJCF."""
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(SOURCE).getroot()
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false",
                  strippath="false")
    for link in robot.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = ROOT / mesh.get("filename").split("/", 3)[3]
            scene = trimesh.load(path, force="scene")
            for index, node in enumerate(scene.graph.nodes_geometry):
                transform, geometry = scene.graph[node]
                part = scene.geometry[geometry].copy()
                part.apply_transform(transform)
                filename = OBJ_DIR / f"{link.get('name')}_{index}.obj"
                part.export(filename, include_texture=False)
                replacement = copy.deepcopy(visual)
                replacement.find("geometry/mesh").set("filename", filename.relative_to(OUTPUT).as_posix())
                material = replacement.find("material")
                if material is not None:
                    replacement.remove(material)
                material = ET.SubElement(replacement, "material", name=f"{link.get('name')}_{index}")
                color = (part.visual.material.main_color if part.visual.kind == "texture"
                         else part.visual.main_color)
                rgba = np.asarray(color, dtype=float) / 255
                ET.SubElement(material, "color", rgba=" ".join(map(str, rgba)))
                link.append(replacement)
            link.remove(visual)
        for mesh in link.findall("collision/geometry/mesh"):
            source = ROOT / mesh.get("filename").split("/", 3)[3]
            target = OBJ_DIR / f"{link.get('name')}_collision.obj"
            trimesh.load(source, force="mesh").export(target)
            mesh.set("filename", target.relative_to(OUTPUT).as_posix())
    converted = OUTPUT / "robot.urdf"
    ET.ElementTree(robot).write(converted, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(converted))
    xml_path = OUTPUT / "scene.xml"
    mujoco.mj_saveLastXML(str(xml_path), model)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 -9.81",
                  integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.03", zfar="50", shadowscale="1")
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        mesh.set("file", "meshes/obj/" + Path(mesh.get("file")).name)
    # Same sky, checker texture, scale and reflection as Kinova_System.xml.
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker",
                  mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.2")
    world = root.find("worldbody")
    for geom in world.iter("geom"):
        if geom.get("mesh", "").endswith("_collision"):
            geom.set("group", "3")
            # STL convex hulls overfill the assembly: enable floor contact only.
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    actuator = ET.SubElement(root, "actuator")
    for joint in world.iter("joint"):
        joint.set("damping", "1.0")
        joint.set("armature", "0.01")
        ET.SubElement(actuator, "motor", name=f"{joint.get('name')}_torque",
                      joint=joint.get("name"), gear="1", ctrllimited="true",
                      ctrlrange=joint.get("actuatorfrcrange"))
    # An oblique spotlight avoids the vertical light's grazing-angle shadow
    # artifacts on the gripper and motor housing while retaining floor shadows.
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 1 -2", directional="false",
                  cutoff="40", castshadow="true",
                  ambient="0.15 0.15 0.15", diffuse="0.9 0.9 0.9", specular="0.3 0.3 0.3")
    # Uniform fill also reaches the distant floor; no extra shadow map means
    # the directional light cannot reintroduce the previous shadow stripes.
    ET.SubElement(world, "light", name="fill", pos="0 0 2", dir="0 0 -1",
                  directional="true", castshadow="false", ambient="0.2 0.2 0.2",
                  diffuse="0.35 0.35 0.35", specular="0 0 0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  pos="0 0 -0.002", material="groundplane", contype="1", conaffinity="2")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=" ".join(map(str, HOME)))
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(xml_path))
