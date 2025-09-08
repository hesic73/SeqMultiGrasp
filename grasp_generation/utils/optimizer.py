import torch

from utils.hand_model import HandModel


class Annealing:
    def __init__(self,
                 hand_model: HandModel,
                 switch_possibility: float = 0.1,
                 starting_temperature: float = 3,
                 temperature_decay: float = 0.95,
                 annealing_period: int = 10,
                 noise_size: float = 0.05,
                 stepsize_period: int = 10,
                 mu: float = 0.98,
                 device='cpu'):
        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(
            starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(
            temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(
            annealing_period, dtype=torch.long, device=device)
        self.noise_size = torch.tensor(
            noise_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(
            stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)
        self.step = 0

        # Initialize old states
        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None

        # EMA grad tracking
        self.ema_grad_hand_pose = torch.zeros(
            self.hand_model.n_dofs + 9, dtype=torch.float, device=device
        )

        # The mask: shape (B, n_dofs+9)
        self.mask = torch.ones(
            *self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device
        )

    def try_step(self, contact_candidates_weight: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            self.mask = mask.to(self.device)
        # Ensure mask shape matches hand_pose shape
        assert self.mask.shape == self.hand_model.hand_pose.shape, "Mask must have shape (B, n_dofs+9)"

        # Compute step size
        s = self.noise_size * self.temperature_decay ** torch.div(
            self.step, self.step_size_period, rounding_mode='floor'
        )
        step_size = torch.zeros(
            *self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device
        ) + s

        # Apply mask to gradients before computing EMA
        masked_grad = self.hand_model.hand_pose.grad * self.mask

        # Update EMA of gradient squares, only considering active DOFs
        self.ema_grad_hand_pose = self.mu * (masked_grad ** 2).mean(0) + \
            (1 - self.mu) * self.ema_grad_hand_pose

        # Update hand pose only on active DOFs
        denom = torch.sqrt(self.ema_grad_hand_pose + 1e-6)
        hand_pose_update = (step_size * masked_grad) / denom.unsqueeze(0)

        hand_pose = self.hand_model.hand_pose - hand_pose_update

        # Switching some contact points at random
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(
            batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()

        weight_matrix = contact_candidates_weight.unsqueeze(
            0).expand(batch_size, -1)

        sampled_indices = torch.multinomial(
            weight_matrix, n_contact, replacement=True)

        contact_point_indices[switch_mask] = sampled_indices[switch_mask]

        # Store old states before committing
        self.old_hand_pose = self.hand_model.hand_pose.clone()
        self.old_contact_point_indices = self.hand_model.contact_point_indices.clone()
        self.old_global_transformation = self.hand_model.global_translation.clone()
        self.old_global_rotation = self.hand_model.global_rotation.clone()
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points.clone()
        self.old_grad_hand_pose = self.hand_model.hand_pose.grad.clone()

        # Set parameters to the new proposed state
        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1

        return s

    def accept_step(self, energy: torch.Tensor, new_energy: torch.Tensor):
        batch_size = energy.shape[0]
        temperature = self.starting_temperature * \
            self.temperature_decay ** torch.div(self.step,
                                                self.annealing_period, rounding_mode='floor')

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        p = torch.exp((energy - new_energy) / temperature)
        accept = alpha < p

        with torch.no_grad():
            reject = ~accept
            # Restore old states for the rejected samples
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]
            self.hand_model.current_status = self.hand_model.chain.forward_kinematics(
                self.hand_model.hand_pose[:, 9:]
            )
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]
            self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature

    def zero_grad(self):
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()

