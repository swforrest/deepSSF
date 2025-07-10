"""
deepSSF_loss.py
==============

Description:
    This script contains the deepSSF model loss function.

Authors:
    Scott Forrest (scottwforrest@gmail.com)
    Dan Pagendam  (dan.pagendam@data61.csiro.au)

Date:
    2025-02-02
    
Usage:
    Run the script as a module, such as in a Jupyter notebook or Python console:
        >>> import deepSSF_model
        >>> deepSSF_model.ConvJointModel(args)
    
    Or execute from the command line:
        $ python deepSSF_model.py [optional arguments]
"""

# Standard library imports
import torch

# Third-party imports
from torch import nn

class negativeLogLikeLoss(nn.Module):
    """
    Custom negative log-likelihood loss that operates on a 4D prediction tensor
    (batch, height, width, channels). The forward pass:
    1. Sums across channel 3 (two log-densities, habitat selection and movement predictions) to obtain a combined log-density.
    2. Multiplies this log-density by the target, which is 0 everywhere except for at the location of the next step, effectively extracting that value,
    then multiplies by -1 such that the function can be minimised (and the probabilities maximised).
    3. Applies the user-specified reduction (mean, sum, or none).
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'mean', 'sum', or 'none'.
        """
        super(negativeLogLikeLoss, self).__init__()
        assert reduction in ['mean', 'median', 'sum', 'none'], \
            "reduction should be 'mean', 'median', 'sum', or 'none'"
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Forward pass of the negative log-likelihood loss.

        Args:
            predict (Tensor): A tensor of shape (B, H, W, 2) with log-densities
                              across two channels to be summed.
            target  (Tensor): A tensor of the same spatial dimensions (B, H, W)
                              indicating where the log-densities should be evaluated.

        Returns:
            Tensor: The computed negative log-likelihood loss. Shape depends on
                    the reduction method.
        """

        habitat_probability_surface = predict[:, :, :, 0] * 1.0
        movement_probability_surface = predict[:, :, :, 1] * 1.0

        # Check for NaNs in the habitat and movement probability surfaces
        if torch.isnan(habitat_probability_surface).any():
            print("NaNs detected in habitat_probability_surface")
        if torch.isnan(movement_probability_surface).any():
            print("NaNs detected in movement_probability_surface")

        # Sum the log-densities from the two channels
        predict_prod = habitat_probability_surface + movement_probability_surface

        # Check for NaNs in the combined predictions
        if torch.isnan(predict_prod).any():
            print("NaNs detected in predict_prod")
            # print("predict_prod:", predict_prod)
            # raise ValueError("NaNs detected in predict_prod") # to stop training if NaNs are found

        # Normalise the next-step log-densities using the log-sum-exp trick
        # predict_prod = predict_prod - torch.logsumexp(predict_prod, dim = (1, 2), keepdim = True)

        # Pull out next step pixels
        px2, py2 = target

        # Compute negative log-likelihood by multiplying log-densities with target
        # and then flipping the sign
        negLogLike = -1.0 * (predict_prod[:, px2[:], py2[:]])

        # Calculate the loss on the habitat selection surface
        habitat_loss = -1 * (habitat_probability_surface[:, py2[:], px2[:]])

        # Calculate the loss on the movement surface
        movement_loss = -1 * (movement_probability_surface[:, py2[:], px2[:]])

        # Apply the specified reduction
        if self.reduction == 'mean':
            return torch.mean(negLogLike), torch.mean(habitat_loss), torch.mean(movement_loss)
        elif self.reduction == 'median':
            return torch.median(negLogLike), torch.median(habitat_loss), torch.median(movement_loss)
        elif self.reduction == 'sum':
            return torch.sum(negLogLike), torch.sum(habitat_loss), torch.sum(movement_loss)
        elif self.reduction == 'none':
            return negLogLike, habitat_loss, movement_loss

        # Default return (though it should never reach here without hitting an if)
        return negLogLike, habitat_loss, movement_loss

