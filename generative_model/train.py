import hydra
from omegaconf import DictConfig, OmegaConf

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler

from diffusers import DDPMScheduler
from tqdm import tqdm

from src.data import MultiGraspDataset
from src.data.utils import make_process_batch_fn, make_x0_normalizer, RotationRepresentation
from src.network.unet import UNetModel
from src.network.model import MyModel
from src.consts import CONFIG_PATH
from src.utils.misc_utils import set_seed, AverageMeter
from src.utils.logger import get_logger, Logger

from loguru import logger as loguru_logger

from typing import Optional, Callable, Any, Dict

_MESHDATA_PATH = str(Path(__file__).parent.parent / "grasp_generation" / "assets" / "objects")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_logger: Optional[Logger] = None,
    process_batch_fn: Optional[Callable[[Any,], Any]] = None,
    device="cuda",
    global_step: int = 0,
    epoch: int = 0,
    rand_t_type: str = "half",
):
    model.train()

    loss_meter = AverageMeter()

    for step, batch in enumerate(dataloader):

        if process_batch_fn is not None:
            batch = process_batch_fn(batch)

        x0: torch.Tensor = batch["x0"]               # [B, 9+9+qpos_dim]
        pc0 = batch["object_0_points"]               # [B, sampled_points, 3]
        pc1 = batch["object_1_points"]               # same as above
        pc: torch.Tensor = torch.stack(
            [pc0, pc1], dim=1)

        x0 = x0.to(device)
        pc = pc.to(device)

        # 1) sample timestep and add noise
        bsz = x0.shape[0]

        if rand_t_type == "all":
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        elif rand_t_type == "half":
            half_bsz = (bsz + 1) // 2
            t_half = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (half_bsz,), device=device).long()

            if bsz % 2 == 1:
                t = torch.cat(
                    [t_half, noise_scheduler.config.num_train_timesteps - t_half[:-1] - 1], dim=0)
            else:
                t = torch.cat(
                    [t_half, noise_scheduler.config.num_train_timesteps - t_half - 1], dim=0)
        else:
            raise ValueError(
                "Unsupported rand_t_type. Choose from ['all', 'half'].")

        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)  # [B, d_x]

        # 2) forward pass
        pred_noise = model(x_t, t, pc)                # [B, d_x]

        # 3) compute loss and optimize
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_step = global_step + step
        if train_logger is not None:
            train_logger.log({
                "train_loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            },
                global_step=current_step)

        loss_meter.update(loss.item())

    return {
        "avg_loss": loss_meter.avg(),
        'global_step': global_step + len(dataloader)
    }


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)
    print("Current Configuration:\n")
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.train.get("seed", 42)
    set_seed(seed)

    run_name = cfg.get("run_name", None)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    logdir = os.path.join(cfg.get("log_root_dir"), run_name, str(seed))
    os.makedirs(logdir, exist_ok=True)
    ckpt_dir = os.path.join(logdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    config_save_path = os.path.join(logdir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = cfg.train.device

    # model
    unet = UNetModel(**cfg.model)
    model = MyModel(unet).to(device)

    # noise_scheduler
    noise_scheduler = DDPMScheduler(**cfg.scheduler)
    rand_t_type = cfg.scheduler.get("rand_t_type", "half")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # dataloader
    dataset = MultiGraspDataset(
        data_path=cfg.data.data_path,
        meshdata_path=_MESHDATA_PATH,
        object_lists=cfg.data.object_lists,
    )
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(
        sampler, batch_size=cfg.train.batch_size, drop_last=True)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # scheduler
    n_total_training_steps = ((len(dataset)//cfg.train.batch_size) *
                              cfg.train.num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_total_training_steps, eta_min=1e-5)

    # logger
    logger_cfg = cfg.logger
    OmegaConf.set_struct(logger_cfg, False)
    if logger_cfg.logger_type == "tensorboard":
        logger_cfg.log_dir = os.path.join(logdir, "tb_logs")
    elif logger_cfg.logger_type == "wandb":
        logger_cfg.name = run_name

    train_logger = get_logger(logger_cfg)

    use_keypoints = cfg.get("use_keypoints", True)
    q_noise_var = cfg.get("q_noise_var", None)
    rot_augmentation = cfg.get("rot_augmentation", False)
    rotation_representation = cfg.get(
        "rotation_representation", None) or "rot6d"
    rotation_representation = RotationRepresentation[rotation_representation]

    if use_keypoints:
        from src.utils.keypoints_utils import HandModelKeypoints
        hand_model_keypoints = HandModelKeypoints(device=device)
    else:
        hand_model_keypoints = None

    stats_path = cfg.get("stats_path", None)

    if stats_path is not None:
        assert os.path.exists(stats_path)
        x0_normalizer = make_x0_normalizer(
            stats_path, use_keypoints, device=device, rotation_representation=rotation_representation)
    else:
        x0_normalizer = None

    process_batch_fn = make_process_batch_fn(
        use_keypoints=use_keypoints,
        q_noise_var=q_noise_var,
        hand_model_keypoints=hand_model_keypoints,
        x0_normalizer=x0_normalizer,
        rot_augmentation=rot_augmentation,
        device=device,
        rotation_representation=rotation_representation
    )

    # training
    num_epochs = cfg.train.num_epochs
    save_interval = cfg.train.get("save_interval", -1)

    if cfg.get("eval", None) is not None:
        eval_interval = cfg.eval.get("eval_interval", -1)
    else:
        eval_interval = -1

    start_epoch = 0
    global_step = 0

    resume_ckpt_path = cfg.train.get("resume_checkpoint", None)
    if resume_ckpt_path:
        if os.path.isfile(resume_ckpt_path):
            loguru_logger.info(f"Resuming from checkpoint: {resume_ckpt_path}")
            ckpt_data = torch.load(resume_ckpt_path, map_location=device)

            model.load_state_dict(ckpt_data["model"])

            resume_model_only = cfg.train.get("resume_model_only", False)

            if not resume_model_only:
                if "optimizer" in ckpt_data and ckpt_data["optimizer"] is not None:
                    optimizer.load_state_dict(ckpt_data["optimizer"])
                if "scheduler" in ckpt_data and ckpt_data["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt_data["scheduler"])

                start_epoch = ckpt_data.get("epoch", 0)
                global_step = ckpt_data.get("global_step", 0)
            else:
                loguru_logger.info("Only model weights are resumed.")

            loguru_logger.info(
                f"Resumed at epoch {start_epoch}, global_step={global_step}")
        else:
            loguru_logger.warning(
                f"resume_checkpoint '{resume_ckpt_path}' not found. Start from scratch.")


    with tqdm(range(start_epoch, num_epochs), desc="Training Progress", unit="epoch") as pbar:
        for epoch in pbar:
            info = train_one_epoch(
                model=model,
                dataloader=dataloader,
                noise_scheduler=noise_scheduler,
                optimizer=optimizer,
                scheduler=scheduler,
                train_logger=train_logger,
                process_batch_fn=process_batch_fn,
                device=device,
                global_step=global_step,
                epoch=epoch + 1,
                rand_t_type=rand_t_type,
            )
            global_step = info['global_step']
            avg_loss = info['avg_loss']

            # update tqdm postfix
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            # if it's time to save checkpoint, save it
            if (save_interval > 0) and ((epoch + 1) % save_interval == 0):
                ckpt_path = os.path.join(ckpt_dir, f"{epoch + 1:04d}.pth")

                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step
                }
                torch.save(ckpt_data, ckpt_path)

                loguru_logger.info(f"Checkpoint saved at: {ckpt_path}")

            if (eval_interval > 0) and ((epoch + 1) % eval_interval == 0):
                model.eval()
                # TODO: eval code here
                model.train()

    loguru_logger.info("Training complete.")
    final_ckpt_path = os.path.join(ckpt_dir, "final.pth")

    ckpt_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": num_epochs,
        "global_step": global_step
    }
    torch.save(ckpt_data, final_ckpt_path)

    print(f"Final checkpoint saved at: {final_ckpt_path}")

    train_logger.close()


if __name__ == "__main__":
    main()