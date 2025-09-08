from utils.hand_model import HandModel
import plotly.graph_objects as go
import transforms3d
import torch
import numpy as np
import json
import plotly.io as pio
pio.renderers.default = "browser"

from src.consts import HAND_URDF_PATH, CONTACT_CANDIDATES_PATH


if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    contact_candidates = json.load(open(CONTACT_CANDIDATES_PATH, 'r'))

    contact_candidates = {
        k: torch.tensor(v, dtype=torch.float, device=device) for k, v in contact_candidates.items()
    }

    hand_model = HandModel(
        urdf_path=HAND_URDF_PATH,
        contact_candidates=contact_candidates,
        n_surface_points=1000,
        device=device
    )
    rot = transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz')
    hand_pose = torch.cat([
        torch.tensor([0, 0, 0], dtype=torch.float, device=device),
        # torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
        torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
        # torch.zeros([16], dtype=torch.float, device=device),
        torch.tensor([
            0, 0.5, 0, 0,
            0, 0.5, 0, 0,
            0, 0.5, 0, 0,
            0.4, 0, 0, 0,
        ], dtype=torch.float, device=device),
    ], dim=0)
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # info
    contact_candidates = hand_model.get_contact_candidates_world()[0]
    surface_points = hand_model.get_surface_points()[0]
    print(f'n_dofs: {hand_model.n_dofs}')
    print(f'n_contact_candidates: {len(contact_candidates)}')
    print(f'n_surface_points: {len(surface_points)}')
    print(hand_model.joint_names)

    # visualize

    hand_plotly = hand_model.get_plotly_data(
        i=0, opacity=0.5, color='lightblue', with_contact_candidates=False)

    object_0_contact_candidates = hand_model.get_contact_candidates_on_links_world(
        ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip'])[0].detach().cpu()
    object_0_contact_candidates_plotly = [go.Scatter3d(
        x=object_0_contact_candidates[:, 0], y=object_0_contact_candidates[:, 1], z=object_0_contact_candidates[:, 2], mode='markers', marker=dict(size=10, color='red'))]

    object_1_contact_candidates = hand_model.get_contact_candidates_on_links_world(
        ["link_9.0", "link_10.0", "link_11.0", "link_11.0_tip", "base_link"])[0].detach().cpu()

    object_1_contact_candidates_plotly = [go.Scatter3d(
        x=object_1_contact_candidates[:, 0], y=object_1_contact_candidates[:, 1], z=object_1_contact_candidates[:, 2], mode='markers', marker=dict(size=10, color='blue'))]

    fig = go.Figure(hand_plotly + object_0_contact_candidates_plotly +
                    object_1_contact_candidates_plotly)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        )
    )

    fig.show()
