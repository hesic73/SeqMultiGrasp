import os

CONFIG_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "config")


ALLEGRO_HAND_URDF_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "robot_models/urdf/allegro_hand_description/allegro_hand_description_right.urdf")


ALLEGRO_HAND_KEYPOINTS_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "robot_models/meta/allegro/keypoints.json")

ALLEGRO_HAND_CONTACT_CANDIDATES_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "robot_models/meta/allegro/contact_candidates.json")
