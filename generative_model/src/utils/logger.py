import abc


from typing import Dict, Any, Optional


class Logger(abc.ABC):
    @abc.abstractmethod
    def log(self, metrics: Dict[str, Any], global_step: Optional[int] = None):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class TensorboardLogger(Logger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: Dict[str, Any], global_step: Optional[int] = None):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, global_step)

    def close(self):
        self.writer.close()


class WandbLogger(Logger):
    def __init__(self, **kwargs):
        import wandb
        wandb.init(**kwargs)
        self.wandb = wandb

    def log(self, metrics: Dict[str, Any], global_step: Optional[int] = None):
        if global_step is not None:
            self.wandb.log({"global_step": global_step, **metrics})
        else:
            self.wandb.log(metrics)

    def close(self):
        self.wandb.finish()


def get_logger(cfg) -> Logger:
    if cfg.logger_type == "tensorboard":
        return TensorboardLogger(cfg.log_dir)
    elif cfg.logger_type == "wandb":
        return WandbLogger(
            project=cfg.get("project", None),
            group=cfg.get("group", None),
            name=cfg.get("name", None),
            mode=cfg.get("mode", None),
            job_type=cfg.get("job_type", None),
        )
    else:
        raise ValueError(f"Unknown logger: {cfg.logger}")
