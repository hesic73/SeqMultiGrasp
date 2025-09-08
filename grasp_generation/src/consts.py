import os

import pathlib


_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()

MESHDATA_PATH = _ROOT_PATH/'data'/'meshdata'

EXPERIMENTS_PATH = _ROOT_PATH/'data'/'experiments'

CONTACT_CANDIDATES_PATH = _ROOT_PATH/'robot_models' / \
    'meta'/'allegro' / 'contact_candidates.json'


HAND_URDF_PATH = _ROOT_PATH/'robot_models'/'urdf'/'allegro_hand_description' / \
    'allegro_hand_description_right.urdf'


HAND_WIDTH_MAPPER_META_PATH = _ROOT_PATH/'robot_models'/'meta'/'allegro' / \
    'width_mapper_meta.yaml'

_ASSET_PATH = pathlib.Path(__file__).parent.parent.absolute()/'assets'


def get_object_mesh_path(object_name: str):
    p = _ASSET_PATH / "objects"/f"{object_name}"/f"{object_name}.stl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return str(p)


CONFIG_PATH = str(pathlib.Path(__file__).parent.parent.absolute()/'config')


_FRICTION_COEFFICIENT = 0.9
_DENSITY = 500
