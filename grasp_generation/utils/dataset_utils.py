import h5py
import numpy as np
from collections import defaultdict


from typing import List, Dict, Any


_thumb_links = ["link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"]
_index_links = ["link_1.0", "link_2.0", "link_3.0", "link_3.0_tip"]
_middle_links = ["link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"]
_ring_links = ["link_9.0", "link_10.0", "link_11.0", "link_11.0_tip"]
_palm_links = ["base_link"]

_thumb_joints = ['joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']
_index_joints = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0']
_middle_joints = ['joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0']
_ring_joints = ['joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0']

_links = ['base_link', 'link_0.0', 'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip', 'link_4.0', 'link_5.0', 'link_6.0', 'link_7.0', 'link_7.0_tip',
          'link_8.0', 'link_9.0', 'link_10.0', 'link_11.0', 'link_11.0_tip', 'link_12.0', 'link_13.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']



def contact_point_indices_to_contact_mask(contact_point_indices: List[int], contact_candidates: Dict[str, List[List[float]]]) -> np.ndarray:
    """convert contact point indices to mask

    Args:
        contact_point_indices (List[int]): contact point indices
        contact_candidates (Dict[str, List[List[float]]]): contact candidates

    Returns:
        np.ndarray: (5,) thumb, index, middle, ring, palm
    """
    # thumb, index, middle, ring, palm
    mask = np.zeros(5, dtype=np.int32)

    idx = 0

    for link_name in _links:
        for _ in contact_candidates[link_name]:
            if idx in contact_point_indices:
                if link_name in _thumb_links:
                    mask[0] = 1
                elif link_name in _index_links:
                    mask[1] = 1
                elif link_name in _middle_links:
                    mask[2] = 1
                elif link_name in _ring_links:
                    mask[3] = 1
                elif link_name in _palm_links:
                    mask[4] = 1
                else:
                    raise ValueError(f"Invalid link_name: {link_name}")
            idx += 1

    return mask


def contact_mask_to_joint_names(mask: np.ndarray) -> List[str]:
    """convert mask to joint names

    Args:
        mask (np.ndarray): (4,)

    Returns:
        List[str]: list of joint names
    """
    joint_names = []
    for i, flag in enumerate(mask[:4]):
        if flag == 1:
            if i == 0:
                joint_names.extend(_thumb_joints)
            elif i == 1:
                joint_names.extend(_index_joints)
            elif i == 2:
                joint_names.extend(_middle_joints)
            elif i == 3:
                joint_names.extend(_ring_joints)
            else:
                raise ValueError(f"Invalid mask: {mask}")

    return joint_names


def contact_mask_to_joint_indices(mask: np.ndarray) -> List[int]:
    indices = []

    for i, flag in enumerate(mask[:4]):
        if flag == 1:
            if i == 0:
                indices.extend([12, 13, 14, 15])
            elif i == 1:
                indices.extend([0, 1, 2, 3])
            elif i == 2:
                indices.extend([4, 5, 6, 7])
            elif i == 3:
                indices.extend([8, 9, 10, 11])
            else:
                raise ValueError(f"Invalid mask: {mask}")

    return indices


def contact_point_index_to_link_name(contact_point_index: int, contact_candidates: Dict[str, List[List[float]]]) -> str:

    # NOTE (hsc): the order matters!

    idx = 0

    for link_name in _links:
        for point in contact_candidates[link_name]:
            if idx == contact_point_index:
                return link_name
            idx += 1

    raise ValueError(f"Invalid contact_point_index: {contact_point_index}")


def link_names_to_mask(link_names: List[str]) -> np.ndarray:

    # thumb, index, middle, ring, palm
    mask = np.zeros(5, dtype=np.int32)
    for link_name in link_names:
        if link_name in _thumb_links:
            mask[0] = 1
        elif link_name in _index_links:
            mask[1] = 1
        elif link_name in _middle_links:
            mask[2] = 1
        elif link_name in _ring_links:
            mask[3] = 1
        elif link_name in _palm_links:
            mask[4] = 1
        else:
            raise ValueError(f"Invalid link_name: {link_name}")

    return mask


def mask_to_full_link_names(mask: np.ndarray) -> List[str]:

    link_names = []
    for i, flag in enumerate(mask):
        if flag == 1:
            if i == 0:
                link_names.extend(_thumb_links)
            elif i == 1:
                link_names.extend(_index_links)
            elif i == 2:
                link_names.extend(_middle_links)
            elif i == 3:
                link_names.extend(_ring_links)
            elif i == 4:
                link_names.extend(_palm_links)
            else:
                raise ValueError(f"Invalid mask: {mask}")

    return link_names
