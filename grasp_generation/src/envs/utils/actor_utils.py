from typing import Optional, Union, Tuple, List, Sequence


import torch
import numpy as np

import os

import sapien
import sapien.physx as physx
from sapien.wrapper.actor_builder import preprocess_mesh_file, do_coacd
from sapien.physx import PhysxMaterial


from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import actors, ActorBuilder

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, to_sapien_pose

import os.path as osp
from pathlib import Path
from transforms3d.euler import euler2quat
from mani_skill.utils.building.ground import build_ground

from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder import table as TABLE
from mani_skill.utils.building.actors.common import _build_by_type

from .actor_builder import VariedActorBuilder


def add_actor_from_file(
        builder: ActorBuilder,
        stl_file_path: str,
        name: str,
        scale: Union[Tuple[float, float, float], float] = 1,
        physical_material: Optional[PhysxMaterial] = None,
        initial_pose: Optional[Pose] = None,
        color=None,
):

    if isinstance(scale, float) or isinstance(scale, int):
        scale = (scale, scale, scale)
    assert isinstance(scale, tuple) and len(scale) == 3

    builder.add_convex_collision_from_file(
        stl_file_path, scale=scale, material=physical_material)

    builder.add_visual_from_file(
        stl_file_path, scale=scale, material=color)

    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)

    return builder.build(name=name)


def add_actor_from_file_with_varied_scale(builder: VariedActorBuilder,
                                          stl_file_path: str,
                                          name: str,
                                          scale: Optional[Sequence[Union[Tuple[float,
                                                                               float, float], float]]] = None,
                                          physical_material: Optional[PhysxMaterial] = None,
                                          initial_pose: Optional[Pose] = None,
                                          color=None,
                                          density: float = 1000):

    builder.add_convex_collision_from_file(
        stl_file_path,  material=physical_material, density=density)

    builder.add_visual_from_file(
        stl_file_path, material=color)

    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)

    return builder.build_with_variation(name=name, scale=scale)


def build_sphere(
    builder: ActorBuilder,
    radius: float,
    color,
    name: str,
    body_type: str = "dynamic",
    physical_material: Optional[PhysxMaterial] = None,
):
    builder.add_sphere_collision(
        radius=radius,
        material=physical_material,
    )
    builder.add_sphere_visual(
        radius=radius,
        material=color,
    )
    return _build_by_type(builder, name, body_type)


def build_cube(
    builder: ActorBuilder,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    physical_material: Optional[PhysxMaterial] = None,
    initial_pose: Optional[Pose] = None,
):
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size] * 3,
            material=physical_material
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=color,
    )
    return _build_by_type(builder, name, body_type, initial_pose=initial_pose)


def build_static_box_with_collision(builder: ActorBuilder,
                                    half_sizes: Sequence[float],
                                    color,
                                    name: str,
                                    initial_pose: Pose,
                                    physical_material: Optional[PhysxMaterial] = None,) -> Actor:
    builder.add_box_collision(
        half_size=half_sizes,
        material=physical_material,
    )
    builder.add_box_visual(
        half_size=half_sizes,
        material=color,
    )
    builder.set_initial_pose(initial_pose)
    return builder.build_static(name=name)


class _TableSceneBuilder(SceneBuilder):
    def build(self, physical_material: Optional[PhysxMaterial] = None, no_visual: bool = False):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(TABLE.__file__)) / "assets"
        table_model_file = os.path.join(model_dir,  "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))

        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
            material=physical_material,
        )
        if not no_visual:
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=table_pose
            )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        table = builder.build_kinematic(name="table-workspace")
        self.table_height = 0.9196429
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -self.table_height],
                        q=euler2quat(0, 0, np.pi / 2))
        )
