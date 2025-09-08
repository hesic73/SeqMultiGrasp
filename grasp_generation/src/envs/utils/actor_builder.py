from typing import Optional, Union, Tuple, List, Sequence


import torch
import numpy as np

import os

import sapien
import sapien.physx as physx
from sapien.wrapper.actor_builder import preprocess_mesh_file, do_coacd
from sapien.physx import PhysxMaterial

from mani_skill import logger
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import actors, ActorBuilder

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, to_sapien_pose

import os.path as osp
from pathlib import Path
from transforms3d.euler import euler2quat
from mani_skill.utils.building.ground import build_ground


class VariedActorBuilder(ActorBuilder):

    def __init__(self):
        super().__init__()
        self._build_with_variation_idx = -1

    def set_mass_and_inertia_with_variation(self, mass: Sequence[float], cmass_local_pose: Sequence[Pose], inertia: Sequence[np.ndarray]):
        self._mass_with_variation = mass
        self._cmass_local_pose_with_variation = cmass_local_pose
        self._inertia_with_variation = inertia
        self._auto_inertial = False
        return self

    def build_render_component_with_variation(self, scale: float = 1.0):
        component = sapien.render.RenderBodyComponent()
        for r in self.visual_records:
            if r.type != "file":
                assert isinstance(r.material, sapien.render.RenderMaterial)
            else:
                assert r.material is None or isinstance(
                    r.material, sapien.render.RenderMaterial
                )

            if r.type == "plane":
                shape = sapien.render.RenderShapePlane(
                    _scale_multiply(r.scale, scale), r.material)
            elif r.type == "box":
                shape = sapien.render.RenderShapeBox(
                    _scale_multiply(r.scale, scale), r.material)
            elif r.type == "sphere":
                shape = sapien.render.RenderShapeSphere(
                    _scale_multiply(r.radius, scale), r.material)
            elif r.type == "capsule":
                shape = sapien.render.RenderShapeCapsule(
                    _scale_multiply(r.radius, scale), _scale_multiply(
                        r.length, scale), r.material
                )
            elif r.type == "cylinder":
                shape = sapien.render.RenderShapeCylinder(
                    _scale_multiply(r.radius, scale), _scale_multiply(
                        r.length, scale), r.material
                )
            elif r.type == "file":
                _scale = _scale_multiply(r.scale, scale)
                shape = sapien.render.RenderShapeTriangleMesh(
                    preprocess_mesh_file(r.filename), _scale, r.material
                )
                if _scale[0] * _scale[1] * _scale[2] < 0:
                    shape.set_front_face("clockwise")
            else:
                raise Exception(f"invalid visual shape type [{r.type}]")

            shape.local_pose = r.pose
            shape.name = r.name
            component.attach(shape)
        return component

    def build_physx_component_with_variation(self, scale: float = 1.0, static_friction: float = None, dynamic_friction: float = None, restitution: float = None, density: float = None, link_parent=None):
        for r in self.collision_records:
            assert isinstance(r.material, physx.PhysxMaterial)

        if self.physx_body_type == "dynamic":
            component = physx.PhysxRigidDynamicComponent()
        elif self.physx_body_type == "kinematic":
            component = physx.PhysxRigidDynamicComponent()
            component.kinematic = True
        elif self.physx_body_type == "static":
            component = physx.PhysxRigidStaticComponent()
        elif self.physx_body_type == "link":
            component = physx.PhysxArticulationLinkComponent(
                link_parent)
        else:
            raise Exception(
                f"invalid physx body type [{self.physx_body_type}]")

        for r in self.collision_records:
            try:
                # Create a new material with adjusted properties
                original_material = r.material
                new_material = physx.PhysxMaterial(
                    static_friction=static_friction if static_friction is not None else original_material.get_static_friction(),
                    dynamic_friction=dynamic_friction if dynamic_friction is not None else original_material.get_dynamic_friction(),
                    restitution=restitution if restitution is not None else original_material.get_restitution()
                )

                # Adjust density if specified
                density_value = density if density is not None else r.density

                if r.type == "plane":
                    # skip adding plane collisions if we already added one.
                    pose_key = (tuple(r.pose.p), tuple(r.pose.q))
                    if (
                        self._allow_overlapping_plane_collisions
                        or pose_key not in self._plane_collision_poses
                    ):
                        shape = physx.PhysxCollisionShapePlane(
                            material=new_material,
                        )
                        shapes = [shape]
                        self._plane_collision_poses.add(pose_key)
                    else:
                        continue
                elif r.type == "box":
                    shape = physx.PhysxCollisionShapeBox(
                        half_size=_scale_multiply(r.scale, scale),
                        material=new_material
                    )
                    shapes = [shape]
                elif r.type == "capsule":
                    shape = physx.PhysxCollisionShapeCapsule(
                        radius=_scale_multiply(r.radius, scale),
                        half_length=_scale_multiply(r.length, scale),
                        material=new_material,
                    )
                    shapes = [shape]
                elif r.type == "cylinder":
                    shape = physx.PhysxCollisionShapeCylinder(
                        radius=_scale_multiply(r.radius, scale),
                        half_length=_scale_multiply(r.length, scale),
                        material=new_material,
                    )
                    shapes = [shape]
                elif r.type == "sphere":
                    shape = physx.PhysxCollisionShapeSphere(
                        radius=_scale_multiply(r.radius, scale),
                        material=new_material,
                    )
                    shapes = [shape]
                elif r.type == "convex_mesh":
                    shape = physx.PhysxCollisionShapeConvexMesh(
                        filename=preprocess_mesh_file(r.filename),
                        scale=_scale_multiply(r.scale, scale),
                        material=new_material,
                    )
                    shapes = [shape]
                elif r.type == "nonconvex_mesh":
                    shape = physx.PhysxCollisionShapeTriangleMesh(
                        filename=preprocess_mesh_file(r.filename),
                        scale=_scale_multiply(r.scale, scale),
                        material=new_material,
                    )
                    shapes = [shape]
                elif r.type == "multiple_convex_meshes":
                    if r.decomposition == "coacd":
                        params = r.decomposition_params
                        if params is None:
                            params = dict()

                        filename = do_coacd(
                            preprocess_mesh_file(r.filename), **params)
                    else:
                        filename = preprocess_mesh_file(r.filename)

                    shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(
                        filename=filename,
                        scale=_scale_multiply(r.scale, scale),
                        material=new_material,
                    )
                else:
                    raise RuntimeError(
                        f"invalid collision shape type [{r.type}]")
            except RuntimeError as e:
                # ignore runtime error (e.g., failed to cook mesh)
                raise e

            for shape in shapes:
                shape.local_pose = r.pose
                shape.set_collision_groups(self.collision_groups)
                shape.set_density(density_value)
                shape.set_patch_radius(r.patch_radius)
                shape.set_min_patch_radius(r.min_patch_radius)
                component.attach(shape)

        if not self._auto_inertial and self.physx_body_type != "kinematic":
            component.mass = self._mass_with_variation[self._build_with_variation_idx]
            component.cmass_local_pose = self._cmass_local_pose_with_variation[
                self._build_with_variation_idx]
            component.inertia = self._inertia_with_variation[self._build_with_variation_idx]

        return component

    def build_entity_with_variation(
        self,
        scale: float = 1.0,
        static_friction: float = None,
        dynamic_friction: float = None,
        restitution: float = None,
        density: float = None,
    ):
        entity = sapien.Entity()
        if self.visual_records:
            entity.add_component(
                self.build_render_component_with_variation(scale=scale))
        entity.add_component(
            self.build_physx_component_with_variation(scale=scale, static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution, density=density))
        entity.name = self.name
        return entity

    def build_kinematic_with_variation(self, name: str, scale: Union[float, Sequence[float]] = None, static_friction: Union[float, Sequence[float]] = None, dynamic_friction: Union[float, Sequence[float]] = None, restitution: Union[float, Sequence[float]] = None, density: Union[float, Sequence[float]] = None):
        self.set_physx_body_type("kinematic")
        return self.build_with_variation(name=name, scale=scale, static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution, density=density)

    def build_static_with_variation(self, name: str, scale: Union[float, Sequence[float]] = None, static_friction: Union[float, Sequence[float]] = None, dynamic_friction: Union[float, Sequence[float]] = None, restitution: Union[float, Sequence[float]] = None, density: Union[float, Sequence[float]] = None):
        self.set_physx_body_type("static")
        return self.build_with_variation(name=name, scale=scale, static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=restitution, density=density)

    def build_with_variation(self, name: str, scale: Union[float, Sequence[float]] = None, static_friction: Union[float, Sequence[float]] = None, dynamic_friction: Union[float, Sequence[float]] = None, restitution: Union[float, Sequence[float]] = None, density: Union[float, Sequence[float]] = None):
        """
        Build the actor with the given name.

        Different to the original SAPIEN API, a unique name is required here.
        """
        self.set_name(name)

        assert (
            self.name is not None
            and self.name != ""
            and self.name not in self.scene.actors
        ), "built actors in ManiSkill must have unique names and cannot be None or empty strings"

        num_actors = self.scene.num_envs
        if self.scene_idxs is not None:
            self.scene_idxs = common.to_tensor(
                self.scene_idxs, device=self.scene.device
            ).to(torch.int)
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)
        num_actors = len(self.scene_idxs)

        if self.initial_pose is None:
            logger.warn(
                f"No initial pose set for actor builder of {self.name}, setting to default pose q=[1,0,0,0], p=[0,0,0]."
            )
            self.initial_pose = Pose.create(sapien.Pose())
        else:
            self.initial_pose = Pose.create(self.initial_pose)

        initial_pose_b = self.initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_actors
        initial_pose_np = common.to_numpy(self.initial_pose.raw_pose)
        if initial_pose_b == 1:
            initial_pose_np = initial_pose_np.repeat(num_actors, axis=0)
        if self.scene.parallel_in_single_scene:
            initial_pose_np[:, :3] += self.scene.scene_offsets_np[
                common.to_numpy(self.scene_idxs)
            ]

        entities = []

        # Helper function to process parameters
        def process_parameter(param, default_value):
            if param is None:
                return [default_value] * num_actors
            elif isinstance(param, (int, float)):
                return [param] * num_actors
            else:
                assert len(
                    param) == num_actors, f"Parameter length does not match number of actors: {len(param)} != {num_actors}"
                return param

        scale = process_parameter(scale, 1.0)
        static_friction = process_parameter(static_friction, None)
        dynamic_friction = process_parameter(dynamic_friction, None)
        restitution = process_parameter(restitution, None)
        density = process_parameter(density, None)

        for i, (scene_idx, _scale, _static_friction, _dynamic_friction, _restitution, _density) in enumerate(zip(self.scene_idxs, scale, static_friction, dynamic_friction, restitution, density)):
            self._build_with_variation_idx = i
            if self.scene.parallel_in_single_scene:
                sub_scene = self.scene.sub_scenes[0]
            else:
                sub_scene = self.scene.sub_scenes[scene_idx]
            entity = self.build_entity_with_variation(
                scale=_scale,
                static_friction=_static_friction,
                dynamic_friction=_dynamic_friction,
                restitution=_restitution,
                density=_density
            )
            # prepend scene idx to entity name to indicate which sub-scene it is in
            entity.name = f"scene-{scene_idx}_{self.name}"
            # set pose before adding to scene
            entity.pose = to_sapien_pose(initial_pose_np[i])
            sub_scene.add_entity(entity)
            entities.append(entity)

        self._build_with_variation_idx = -1
        actor = Actor.create_from_entities(
            entities, self.scene, self.scene_idxs)

        # if it is a static body type and this is a GPU sim but we are given a single initial pose, we repeat it for the purposes of observations
        if (
            self.physx_body_type == "static"
            and initial_pose_b == 1
            and physx.is_gpu_enabled()
        ):
            actor.initial_pose = Pose.create(
                self.initial_pose.raw_pose.repeat(num_actors, 1)
            )
        else:
            actor.initial_pose = self.initial_pose
        self.scene.actors[self.name] = actor
        self.scene.add_to_state_dict_registry(actor)
        return actor


def _scale_multiply(value: Union[Tuple[float, float, float], float], scale: float):
    if isinstance(value, (int, float)):
        return value * scale
    return tuple([v * scale for v in value])
