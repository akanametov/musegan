"""Loss function and gradient penalty for MuseGAN."""

from torch import Tensor

import torch
from torch import nn


class WassersteinLoss(nn.Module):
    """WassersteinLoss."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def forward(self, y_pred: Tensor, y_target: Tensor) -> Tensor:
        """Calculate Wasserstein loss.

        Parameters
        ----------
        y_pred: Tensor
            Prediction.
        y_target: Tensor
            Target.

        Returns
        -------
        Tensor:
            Loss value.

        """
        loss = - torch.mean(y_pred * y_target)
        return loss


class GradientPenalty(nn.Module):
    """Gradient penalty."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def forward(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        """Calculate gradient penalty.

        Parameters
        ----------
        inputs: Tensor
            Input from which to track gradient.
        outputs: Tensor
            Output to which to track gradient.

        Returns
        -------
        Tensor:
            Penalty value.

        """
        grad = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_ = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        penalty = torch.mean((1. - grad_) ** 2)
        return penalty
