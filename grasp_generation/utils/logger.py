from torch.utils.tensorboard import SummaryWriter
import torch


class Logger:
    def __init__(self, log_dir, thres_fc: float = 0.3, thres_dis: float = 0.005, thres_pen: float = 0.02, use_writer: bool = True):
        self.use_writer = use_writer
        self.thres_fc = thres_fc
        self.thres_dis = thres_dis
        self.thres_pen = thres_pen
        if self.use_writer:
            self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, energy: torch.Tensor, E_fc: torch.Tensor, E_dis: torch.Tensor, E_pen: torch.Tensor, E_spen: torch.Tensor, E_joints: torch.Tensor, step: int, show: bool = False):
        success_fc = E_fc < self.thres_fc
        success_dis = E_dis < self.thres_dis
        success_pen = E_pen < self.thres_pen
        success = success_fc * success_dis * success_pen

        if self.use_writer:
            self.writer.add_scalar('Energy/energy', energy.mean(), step)
            self.writer.add_scalar('Energy/fc', E_fc.mean(), step)
            self.writer.add_scalar('Energy/dis', E_dis.mean(), step)
            self.writer.add_scalar('Energy/pen', E_pen.mean(), step)
            self.writer.add_scalar('Energy/spen', E_spen.mean(), step)
            self.writer.add_scalar('Energy/joints', E_joints.mean(), step)

            self.writer.add_scalar(
                'Success/success', success.float().mean(), step)
            self.writer.add_scalar(
                'Success/fc', success_fc.float().mean(), step)
            self.writer.add_scalar(
                'Success/dis', success_dis.float().mean(), step)
            self.writer.add_scalar(
                'Success/pen', success_pen.float().mean(), step)

        if show and step % 100 == 0:
            print(f'Step {step} energy: {energy.mean():.4f}  fc: {E_fc.mean():.4f}  dis: {E_dis.mean():.4f}  pen: {E_pen.mean():.4f}  spen: {E_spen.mean():.4f}  joints: {E_joints.mean():.4f}')
            print(f'success: {success.float().mean():.4f}  fc: {success_fc.float().mean():.4f}  dis: {success_dis.float().mean():.4f}  pen: {success_pen.float().mean():.4f}')

    def close(self):
        if self.use_writer:
            self.writer.close()
