import numpy as np
import torch

from typing import Dict, List, Tuple

from mani_skill.envs.scene import ManiSkillScene
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.base_agent import DictControllerConfig

from mani_skill.utils.structs import Link,  Actor, SimConfig, GPUMemoryConfig, SceneConfig, Pose


from mani_skill.agents.controllers import PDEEPoseControllerConfig, deepcopy_dict, PDJointPosControllerConfig
import os


from src.consts import _ASSET_PATH


@register_agent()
class FrankaAllegoRight(BaseAgent):
    uid = "franka_allegro_right"
    urdf_path = os.path.join(
        _ASSET_PATH, "urdf/franka_allegro_right/franka_allegro_right.urdf")

    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0,
                     dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    # urdf_config = dict(
    #     _materials=dict(
    #         default=dict(static_friction=0.2,
    #                      dynamic_friction=0.2, restitution=0.0)
    #     ),
    #     link={
    #         "link_3.0_tip": dict(
    #             material="default", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #         "link_7.0_tip": dict(
    #             material="default", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #         "link_11.0_tip": dict(
    #             material="default", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #         "link_15.0_tip": dict(
    #             material="default", patch_radius=0.1, min_patch_radius=0.1
    #         ),
    #         "base_link": dict(material="default"),
    #         "link_0.0": dict(material="default"),
    #         "link_1.0": dict(material="default"),
    #         "link_2.0": dict(material="default"),
    #         "link_3.0": dict(material="default"),
    #         "link_4.0": dict(material="default"),
    #         "link_5.0": dict(material="default"),
    #         "link_6.0": dict(material="default"),
    #         "link_7.0": dict(material="default"),
    #         "link_8.0": dict(material="default"),
    #         "link_9.0": dict(material="default"),
    #         "link_10.0": dict(material="default"),
    #         "link_11.0": dict(material="default"),
    #         "link_12.0": dict(material="default"),
    #         "link_13.0": dict(material="default"),
    #         "link_14.0": dict(material="default"),
    #         "link_15.0": dict(material="default"),
    #     },
    # )

    # for debug
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [-0.00946622,  0.03299681,  0.00727816, -2.36426973, -0.00977371,
                 2.38868165,  1.52809203]
                + [0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 1.,
                   0., 0., 0.]
            ),
            pose=sapien.Pose(p=np.array([-0.5, 0.0, 0.0])),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    qmin_arm = torch.tensor(
        [
            -1.5,
            -1.0,
            -2.0,  # -1.0
            -3.0718,
            -2.0,  # -1.0
            -0.0175,
            -2.8973,  # 0.0
        ])
    qmax_arm = torch.tensor(
        [
            1.0,
            1.0,
            2.0,  # 1.0
            -0.0698,
            2.0,  # 1.0
            3.7525,
            2.8973,
        ])

    qmin_hand = torch.tensor([-0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270,
                              -0.4700, -0.1960, -0.1740, -0.2270,  0.2630, -0.1050, -0.1890, -0.1620])
    qmax_hand = torch.tensor([0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180, 0.4700,
                              1.6100, 1.7090, 1.6180, 1.3960, 1.1630, 1.6440, 1.7190])

    hand_joint_names = [
        'joint_0.0',
        'joint_1.0',
        'joint_2.0',
        'joint_3.0',
        'joint_4.0',
        'joint_5.0',
        'joint_6.0',
        'joint_7.0',
        'joint_8.0',
        'joint_9.0',
        'joint_10.0',
        'joint_11.0',
        'joint_12.0',
        'joint_13.0',
        'joint_14.0',
        'joint_15.0',
    ]

    _finger_contact_link_names = [
        "link_15.0_tip", "link_15.0", "link_14.0",  # thumb
        "link_3.0_tip", "link_3.0", "link_2.0", "link_1.0",  # index
        "link_7.0_tip", "link_7.0", "link_6.0", "link_5.0",  # middle
        "link_11.0_tip", "link_11.0", "link_10.0", "link_9.0",  # ring
    ]

    # NOTE (hsc): 基类里palm_link_name是palm
    _palm_link_name = "base_link"

    @property
    def _finger_contact_links(self) -> List[Link]:
        return [self.robot.find_link_by_name(name) for name in self._finger_contact_link_names]

    @property
    def _palm_link(self) -> Link:
        return self.robot.find_link_by_name(self._palm_link_name)

    def _get_net_boolean_contact(self, impulse_threshold: float = 1e-2) -> torch.Tensor:
        t = []
        for link in self._finger_contact_links:
            impulse = link.get_net_contact_forces()
            t.append(torch.norm(impulse, dim=-1) > impulse_threshold)

        t.append(torch.norm(self._palm_link.get_net_contact_forces(),
                 dim=-1) > impulse_threshold)
        return torch.stack(t, dim=-1)  # (n, 16)

    def _get_boolean_contact(
        self,
        scene: ManiSkillScene,
        obj: Actor,
        impulse_threshold: float = 1e-2,
    ) -> torch.Tensor:
        t = []
        for link in self._finger_contact_links:
            impulse = scene.get_pairwise_contact_impulses(obj, link)
            t.append(torch.norm(impulse, dim=-1) > impulse_threshold)

        impulse = scene.get_pairwise_contact_impulses(obj, self._palm_link)
        t.append(torch.norm(impulse, dim=-1) > impulse_threshold)
        return torch.stack(t, dim=-1)  # (n, 16)

    def __init__(self,
                 scene: ManiSkillScene,
                 control_freq: int,
                 control_mode: str = None,
                 agent_idx: int = None,
                 initial_pose=None,
                 *args, **kwargs):

        self.arm_joint_pos_stiffness = 1000
        self.arm_joint_pos_damping = 100

        self.arm_force_limit = 100

        self.hand_stiffness = 4e2
        self.hand_damping = 1e1
        self.hand_force_limit = 5e1

        self.ee_link_name = 'base_link'

        super().__init__(scene, control_freq, control_mode, agent_idx,
                         initial_pose, *args, **kwargs)

    def _make_pd_joint_pos_hand_config(self):

        # hand_pd_pos = PDJointPosControllerConfig(
        #     self.hand_joint_names,
        #     self.qmin_hand,
        #     self.qmax_hand,
        #     self.hand_stiffness,
        #     self.hand_damping,
        #     self.hand_force_limit,
        #     use_delta=False,
        # )

        hand_pd_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            None,
            None,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=False,
            normalize_action=False,
        )
        return hand_pd_pos

    def _make_pd_joint_pos_config(self):
        """
        Absolute joint position control for both arm and hand
        """

        # arm_pd_pos = PDJointPosControllerConfig(
        #     self.arm_joint_names,
        #     self.qmin_arm,
        #     self.qmax_arm,
        #     self.arm_joint_pos_stiffness,
        #     self.arm_joint_pos_damping,
        #     self.arm_force_limit,
        #     use_delta=False,
        # )

        arm_pd_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_joint_pos_stiffness,
            self.arm_joint_pos_damping,
            self.arm_force_limit,
            use_delta=False,
            normalize_action=False,
        )

        hand_pd_pos = self._make_pd_joint_pos_hand_config()

        return dict(
            arm=arm_pd_pos, gripper=hand_pd_pos
        )

    @property
    def _controller_configs(
        self,
    ) -> Dict[str,  DictControllerConfig]:

        controller_configs = dict(
            pd_joint_pos=self._make_pd_joint_pos_config(),
        )

        return deepcopy_dict(controller_configs)
