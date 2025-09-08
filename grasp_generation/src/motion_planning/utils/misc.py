from typing import List

from curobo.types.math import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
import sapien


def allegro_hand_reorder_joint_indices(joint_names: List[str]) -> List[int]:
    # 提取 joint 名称中的索引部分并记录其原始索引
    indexed_joints = [(name, i) for i, name in enumerate(joint_names)]

    # 根据 joint 名称中的数字进行排序
    sorted_joints = sorted(indexed_joints, key=lambda x: int(
        x[0].split('_')[1].split('.')[0]))

    # 返回重新排序后的索引列表
    reordered_indices = [index for _, index in sorted_joints]

    return reordered_indices


def sapien_pose_to_curobo_pose(pose: sapien.Pose, tensor_args: TensorDeviceType = TensorDeviceType()) -> CuroboPose:
    l = list(pose.p)+list(pose.q)
    return CuroboPose.from_list(l, tensor_args=tensor_args)
