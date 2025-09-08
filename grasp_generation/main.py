from omegaconf import OmegaConf
import hydra
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.logger import Logger
from utils.optimizer import Annealing
from utils.energy import cal_energy
from utils.initializations import (
    single_initialize_convex_hull,
    initialize_multi_grasp,
    initialize_side_grasp,
)
from utils.object_model import ObjectModel
from utils.single_object_model import SingleObjectModel
from utils.hand_model import HandModel

from utils.misc import (
    make_contact_candidates_weight,
    hand_pose_to_dict,
    extract_hand_pose_and_scale_tensor,
    get_joint_positions_from_qpos,
)
from utils.validation import validate_grasp

from src.validate_tabletop import validate_one_object_tabletop
from src.consts import HAND_URDF_PATH, CONTACT_CANDIDATES_PATH, MESHDATA_PATH, EXPERIMENTS_PATH

import torch
import numpy as np
import shutil
import os
import json
import csv

from tqdm import tqdm
from loguru import logger as loguru_logger

from typing import List, Dict


@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))

    np.seterr(all='raise')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    loguru_logger.info(f'Random seed: {cfg.seed}')

    if cfg.get('gpu', None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
        assert torch.cuda.is_available(), 'GPU is not available'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        loguru_logger.warning('Running on CPU')

    contact_candidates = json.load(open(CONTACT_CANDIDATES_PATH, 'r'))

    active_links = cfg.active_links
    loguru_logger.info(f'Active links: {active_links}')

    contact_candidates_weight = make_contact_candidates_weight(
        contact_candidates=contact_candidates,
        active_links=active_links,
        link_base_weight={'base_link': 3.0}
    ).to(device)

    contact_candidates = {
        k: torch.tensor(v, dtype=torch.float, device=device)
        for k, v in contact_candidates.items()
    }

    hand_model = HandModel(
        urdf_path=HAND_URDF_PATH,
        contact_candidates=contact_candidates,
        n_surface_points=1000,
        device=device
    )

    object_model = SingleObjectModel(
        data_root_path=MESHDATA_PATH,
        batch_size=cfg.batch_size,
        num_samples=2000,
        device=device
    )
    object_model.initialize(cfg.object_code)
    total_batch_size = cfg.batch_size

    if cfg.initialization_method == "convex_hull":
        single_initialize_convex_hull(
            hand_model=hand_model,
            object_model=object_model,
            distance_lower=0.2,
            distance_upper=0.3,
            theta_lower=-np.pi / 6,
            theta_upper=np.pi / 6,
            jitter_strength=0.1,
            n_contact=cfg.n_contact,
            contact_candidates_weight=contact_candidates_weight,
        )

    elif cfg.initialization_method == "multi_grasp":
        initialize_multi_grasp(
            hand_model=hand_model,
            batch_size=cfg.batch_size,
            n_contact=cfg.n_contact,
            contact_candidates_weight=contact_candidates_weight,
            qpos_loc=cfg.qpos_loc,
            jitter_strength=cfg.jitter_strength,
        )
    elif cfg.initialization_method == "side_grasp":
        initialize_side_grasp(
            hand_model=hand_model,
            batch_size=cfg.batch_size,
            n_contact=cfg.n_contact,
            contact_candidates_weight=contact_candidates_weight,
            qpos_loc=cfg.qpos_loc,
            jitter_strength=cfg.jitter_strength,
        )
    else:
        raise ValueError(f"Invalid initialization method: {cfg.initialization_method}. Valid options are: 'convex_hull', 'multi_grasp', 'side_grasp'")

    loguru_logger.info(
        f'Number of contact candidates: {hand_model.n_contact_candidates}')
    loguru_logger.info(f'Total batch size: {total_batch_size}')

    # Record the initial hand pose and contact indices
    hand_pose_st = hand_model.hand_pose.detach().clone()
    contact_point_indices_st = hand_model.contact_point_indices.clone()

    optim_config = {
        'switch_possibility': cfg.optimizer.switch_possibility,
        'starting_temperature': cfg.optimizer.starting_temperature,
        'temperature_decay': cfg.optimizer.temperature_decay,
        'annealing_period': cfg.optimizer.annealing_period,
        'noise_size': cfg.optimizer.noise_size,
        'stepsize_period': cfg.optimizer.stepsize_period,
        'mu': cfg.optimizer.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    log_dir = str(EXPERIMENTS_PATH/cfg.name/'logs')
    if os.path.exists(log_dir):
        loguru_logger.warning(
            f'log_dir {log_dir} already exists, removing it...')
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    logger_config = {
        'thres_fc': cfg.thres_fc,
        'thres_dis': cfg.thres_dis,
        'thres_pen': cfg.thres_pen,
        'use_writer': cfg.use_writer,
    }
    logger = Logger(log_dir=log_dir, **logger_config)

    weight_dict = dict(
        w_dis=cfg.w_dis,
        w_pen=cfg.w_pen,
        w_spen=cfg.w_spen,
        w_joints=cfg.w_joints,
        levitate=cfg.levitate,
        ground_offset=cfg.ground_offset,
        object_min_y=object_model.object_min_y
    )
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(
        hand_model, object_model, verbose=True, **weight_dict
    )


    energy.sum().backward(retain_graph=True)

    logger.log(energy, E_fc, E_dis, E_pen, E_spen,
               E_joints, step=0, show=cfg.verbose)

    # ------------------ Annealing loop ------------------
    for step in tqdm(range(1, cfg.n_iter + 1), desc='Annealing', disable=cfg.get('disable_tqdm', False)):

        optimizer.try_step(contact_candidates_weight=contact_candidates_weight)

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(
            hand_model, object_model, verbose=True, **weight_dict
        )
        new_energy.sum().backward(retain_graph=True)

        # Decide whether to accept this new state
        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]

            logger.log(energy, E_fc, E_dis, E_pen,
                       E_spen, E_joints, step, show=cfg.verbose)


    # ------------------ Save results ------------------
    result_path = str(EXPERIMENTS_PATH/cfg.name/'results')
    if os.path.exists(result_path):
        loguru_logger.warning(
            f'result_path {result_path} already exists, removing it...')
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    # Save per-object results
    data_list = []
    for idx in range(cfg.batch_size):
        scale = object_model.scale_factor[idx].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        qpos = hand_pose_to_dict(hand_pose)
        contact_point_indices_i = (
            hand_model.contact_point_indices[idx].detach(
            ).cpu().numpy().tolist()
        )
        hand_pose_st_i = hand_pose_st[idx].detach().cpu()
        qpos_st_i = hand_pose_to_dict(hand_pose_st_i)
        contact_point_indices_st_i = contact_point_indices_st[idx].cpu(
        ).numpy().tolist()

        data_list.append(dict(
            object_code=cfg.object_code,
            scale=scale,
            qpos=qpos,
            contact_point_indices=contact_point_indices_i,
            qpos_st=qpos_st_i,
            contact_point_indices_st=contact_point_indices_st_i,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
        ))

    np.save(
        os.path.join(result_path, f"{cfg.object_code}.npy"),
        data_list,
        allow_pickle=True
    )

    # Extract successful pre-grasp poses
    success_list = extract_successful_pre_grasp_poses_one_object(
        os.path.join(result_path, f"{cfg.object_code}.npy"),
        thres_fc=cfg.thres_fc,
        thres_dis=cfg.thres_dis,
        thres_pen=cfg.thres_pen,
    )
    success_list = np.array(success_list)

    # Save successful poses
    dfc_success_data_path = os.path.join(
        result_path, f"{cfg.object_code}_success.npy")
    np.save(
        dfc_success_data_path,
        success_list,
        allow_pickle=True
    )

    n_success = len(success_list)

    # Validate the successful poses
    validated_data_path = os.path.join(
        result_path, f"{cfg.object_code}_success_validated.npy")
    validate_grasp(dfc_success_data_path, validated_data_path)

    n_validated = len(np.load(validated_data_path, allow_pickle=True))

    # execution-based validation
    tabletop_success_indices = validate_one_object_tabletop(
        validated_data_path)
    tabletop_validated_data_path = os.path.join(
        result_path, f"{cfg.object_code}_tabletop_validated.npy")
    np.save(
        tabletop_validated_data_path,
        np.load(validated_data_path, allow_pickle=True)[
            tabletop_success_indices],
        allow_pickle=True
    )

    n_tabletop_validated = len(tabletop_success_indices)
    synthesize_success_rate = n_success/cfg.batch_size
    validation_success_rate = n_validated/n_success if n_success > 0 else 0.0
    tabletop_validation_success_rate = n_tabletop_validated / \
        n_validated if n_validated > 0 else 0.0
    cumulative_success_rate = synthesize_success_rate * \
        validation_success_rate*tabletop_validation_success_rate

    # Prepare data for CSV
    metrics = {
        "seed": cfg.seed,
        "object_code": cfg.object_code,
        "batch_size": cfg.batch_size,
        "number of successful grasps": n_success,
        "number of validated grasps": n_validated,
        "number of tabletop validated grasps": n_tabletop_validated,
        "synthesize_success_rate": synthesize_success_rate,
        "validation_success_rate": validation_success_rate,
        "tabletop_validation_success_rate": tabletop_validation_success_rate,
        "cumulative_success_rate": cumulative_success_rate
    }

    # Define file path
    csv_file_path = os.path.join(result_path, "info.csv")

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(metrics.keys())
        # Write values
        writer.writerow(metrics.values())

    print(
        f"#DFC Success:{n_success:04d}\t #Validated:{n_validated:04d}\t #Tabletop Validated:{n_tabletop_validated:04d}")
    print(
        f"Success rate: {cumulative_success_rate} ({n_tabletop_validated}/{cfg.batch_size})")


def extract_successful_pre_grasp_poses_one_object(path: str, thres_fc: float, thres_dis: float, thres_pen: float) -> List[Dict]:
    l = np.load(path, allow_pickle=True)

    def _is_successful(E_fc: float, E_dis: float, E_pen: float) -> bool:
        return E_fc < thres_fc and E_dis < thres_dis and E_pen < thres_pen

    successful_list = []

    for idx, item in enumerate(l):
        object_code: str = item['object_code']
        scale: float = item['scale']
        qpos: Dict[str, float] = item['qpos']
        contact_point_indices: List[int] = item['contact_point_indices']

        E_fc: float = item['E_fc']
        E_dis: float = item['E_dis']
        E_pen: float = item['E_pen']
        E_spen: float = item['E_spen']
        E_joints: float = item['E_joints']

        if _is_successful(E_fc, E_dis, E_pen):

            # xyz = np.array([qpos[name] for name in _translation_names])
            # rpy = np.array([qpos[name] for name in _rot_names])
            # hand_qpos = np.array([qpos[name] for name in _joint_names])

            successful_list.append({
                'object_code': object_code,
                'scale': scale,
                'qpos': qpos,
                # 'xyz': xyz,
                # 'rpy': rpy,
                # 'hand_qpos': hand_qpos,
                'contact_point_indices': contact_point_indices,
                'original_idx': idx,
            })

    return successful_list


if __name__ == '__main__':
    main()