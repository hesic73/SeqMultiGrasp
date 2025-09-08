from mani_skill.agents.robots.allegro_hand import AllegroHandRight as _AllegroHandRight
from mani_skill.agents.registration import register_agent


from mani_skill.utils import sapien_utils

from mani_skill.utils.structs import Link, Actor
from mani_skill.envs.scene import ManiSkillScene

import torch

from typing import List, Tuple, Union, Dict, Any, Optional, Sequence


@register_agent(override=True)
class AllegroHandRight(_AllegroHandRight):
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

    _finger_contact_link_names = [
        "link_15.0_tip", "link_15.0", "link_14.0",  # thumb
        "link_3.0_tip", "link_3.0", "link_2.0", "link_1.0",  # index
        "link_7.0_tip", "link_7.0", "link_6.0", "link_5.0",  # middle
        "link_11.0_tip", "link_11.0", "link_10.0", "link_9.0",  # ring
    ]

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
