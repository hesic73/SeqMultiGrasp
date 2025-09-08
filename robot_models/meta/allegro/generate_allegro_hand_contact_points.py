
"""
Reference:
    https://github.com/tylerlum/get_a_grip/blob/main/get_a_grip/dataset_generation/scripts/generate_allegro_contact_points_precision_grasp.py
"""
import json

import numpy as np
import plotly.graph_objects as go

import os


def make_fingertip_contact_points():

    phi_min = -0.6 * np.pi / 2.0 
    phi_max = 0.6 * np.pi / 2.0
    theta_min = 0.15 * np.pi 
    theta_max = 0.45 * np.pi 

    N_theta = 3
    N_phi_list = [6, 5, 4]

    if len(N_phi_list) != N_theta:
        raise ValueError("Length of N_phi_list must match N_theta.")

    cos_theta_min = np.cos(theta_min)
    cos_theta_max = np.cos(theta_max)

    cos_thetas = np.linspace(cos_theta_max, cos_theta_min, N_theta)

    thetas = np.arccos(cos_thetas)

    points = []

    for i, theta in enumerate(thetas):
        N_phi = N_phi_list[i]
        phis = np.linspace(phi_min, phi_max, N_phi)
        x = np.sin(theta) * np.cos(phis)
        y = np.sin(theta) * np.sin(phis)
        z = np.cos(theta)

        for j in range(N_phi):
            points.append([x[j], y[j], z])

    radius = 0.012
    points = np.array(points) * radius

    return points.tolist()


def make_palm_contact_points():

    x_val = 0.0111

    N_yA = 4
    N_zA = 2
    yA_min, yA_max = -0.04, 0.04
    zA_min, zA_max = -0.015, -0.035

    N_yB = 2
    N_zB = 2
    yB_min, yB_max = -0.04, -0.01
    zB_min, zB_max = -0.055, -0.075


    yA_vals = np.linspace(yA_min, yA_max, N_yA)
    zA_vals = np.linspace(zA_min, zA_max, N_zA)
    YA, ZA = np.meshgrid(yA_vals, zA_vals)

    rectA_points = np.stack([
        np.full(YA.size, x_val), 
        YA.ravel(),
        ZA.ravel()
    ], axis=-1)

    yB_vals = np.linspace(yB_min, yB_max, N_yB)
    zB_vals = np.linspace(zB_min, zB_max, N_zB)
    YB, ZB = np.meshgrid(yB_vals, zB_vals) 
    rectB_points = np.stack([
        np.full(YB.size, x_val),
        YB.ravel(),
        ZB.ravel()
    ], axis=-1)

    points = np.concatenate([rectA_points, rectB_points], axis=0)

    return points.tolist()


def main():

    fingertip_points = make_fingertip_contact_points()

    contact_point_dictionary = {
        f"link_{num}.0_tip": fingertip_points for num in [3, 7, 11, 15]
    }

    palm_points = make_palm_contact_points()

    contact_point_dictionary["base_link"] = palm_points

    for i in (3, 7, 11):
        contact_point_dictionary[f"link_{i}.0"] = [
            [0.0098, 0, 0.01335],
            [0.0098, -0.004583333333333333, 0.0089],
            [0.0098, -0.004583333333333333, 0.0178],
            [0.0098, 0.004583333333333333, 0.0089],
            [0.0098, 0.004583333333333333, 0.0178],
        ]

    for i in (2, 6, 10):
        contact_point_dictionary[f"link_{i}.0"] = [
            [0.0098, -0.004583333333333333, 0.0096],
            [0.0098, -0.004583333333333333, 0.0192],
            [0.0098, -0.004583333333333333, 0.0288],
            [0.0098, 0.004583333333333333, 0.0096],
            [0.0098, 0.004583333333333333, 0.0192],
            [0.0098, 0.004583333333333333, 0.0288],
        ]

    for i in (1, 5, 9):
        contact_point_dictionary[f"link_{i}.0"] = [
            [0.0098, -0.004583333333333333, 0.0108],
            [0.0098, -0.004583333333333333, 0.0216],
            [0.0098, -0.004583333333333333, 0.0324],
            [0.0098, -0.004583333333333333, 0.0432],
            [0.0098, 0.004583333333333333, 0.0108],
            [0.0098, 0.004583333333333333, 0.0216],
            [0.0098, 0.004583333333333333, 0.0324],
            [0.0098, 0.004583333333333333, 0.0432],
        ]

    contact_point_dictionary[f"link_13.0"] = []

    contact_point_dictionary[f"link_14.0"] = [
        [0.0098, -0.004583333333333333, 0.010280000000000001],
        [0.0098, -0.004583333333333333, 0.020560000000000002],
        [0.0098, -0.004583333333333333, 0.030840000000000003],
        [0.0098, -0.004583333333333333,  0.041120000000000004],
        [0.0098, 0.004583333333333333, 0.010280000000000001],
        [0.0098, 0.004583333333333333, 0.020560000000000002],
        [0.0098, 0.004583333333333333, 0.030840000000000003],
        [0.0098, 0.004583333333333333, 0.041120000000000004],
    ]

    contact_point_dictionary[f"link_15.0"] = [
        [0.0098, -0.004583333333333333, 0.010575],
        [0.0098, -0.004583333333333333, 0.02115],
        [0.0098, -0.004583333333333333,  0.031724999999999996],
        [0.0098, 0.004583333333333333, 0.010575],
        [0.0098, 0.004583333333333333, 0.02115],
        [0.0098, 0.004583333333333333,   0.031724999999999996],
    ]

    for i in (0, 4, 8, 12):
        contact_point_dictionary[f"link_{i}.0"] = []

    # NOTE: the order matters here! need to match the order in the hand model

    links = ['base_link', 'link_0.0', 'link_1.0', 'link_2.0', 'link_3.0', 'link_3.0_tip', 'link_4.0', 'link_5.0', 'link_6.0', 'link_7.0', 'link_7.0_tip',
             'link_8.0', 'link_9.0', 'link_10.0', 'link_11.0', 'link_11.0_tip', 'link_12.0', 'link_13.0', 'link_14.0', 'link_15.0', 'link_15.0_tip']

    contact_point_dictionary = {k: contact_point_dictionary[k] for k in links}

    if os.path.exists("contact_candidates.json"):
        input("contact_candidates.json already exists. Press Enter to overwrite.")

    with open(
        "contact_candidates.json",
        "w",
    ) as f:
        json.dump(contact_point_dictionary, f)


if __name__ == "__main__":
    main()
