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
from deepSSF_model_polar_movement_fcn_mixture import Params_to_Density_Block

class negativeLogLikeLoss(nn.Module):
    """
    Custom negative log-likelihood loss that operates on a 4D prediction tensor
    (batch, height, width, channels). The forward pass:
    1. Sums across channel 3 (two log-densities, habitat selection and movement predictions) to obtain a combined log-density.
    2. Multiplies this log-density by the target, which is 0 everywhere except for at the location of the next step, effectively extracting that value,
    then multiplies by -1 such that the function can be minimised (and the probabilities maximised).
    3. Applies the user-specified reduction (mean, sum, or none).
    """

    def __init__(self, reduction='mean', params=None):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'mean', 'sum', or 'none'.
        """
        super(negativeLogLikeLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], \
            "reduction should be 'mean', 'sum', or 'none'"
        self.reduction = reduction
        
        self.params = params
        # Set the device for the model
        self.device = params.device

        # Initialize the density calculation block
        self.movement_density_block = Params_to_Density_Block(params).to(self.device)

        # Define weights for habitat vs movement components (can be adjusted)
        self.habitat_loss_weight = 1.0
        self.movement_loss_weight = 1.0

    def forward(self, model_outputs, targets):
        """
        Forward pass of the negative log-likelihood loss.

        Args:
            model_outputs: Tuple of tensors (habitat_output, movement_params, bearing)
            targets: Tuple of tensors (habitat_target, step_length, turn_angle)

        Returns:
            Tensor: The computed (combined and possibly weighted) negative log-likelihood loss. 
            Shape depends on the reduction method.
        """

        # Unpack model outputs
        habitat_output, movement_params, bearing = model_outputs

        # Unpack targets
        habitat_target, step_length, turn_angle = targets

        # Check for NaNs in the combined predictions
        if torch.isnan(habitat_target).any():
            print("NaNs detected in habitat_target")
            print("total_lass:", habitat_target)
            raise ValueError("NaNs detected in habitat_target")

        # Compute negative log-likelihood of the habitat by multiplying log-density with target
        # and then flipping the sign
        habitat_loss = -1 * (habitat_output * habitat_target)
        # print("habitat_loss shape:", habitat_loss.shape)
        habitat_loss = habitat_loss.sum(dim=(1, 2))
        # print("habitat_loss shape:", habitat_loss)

        # Calculate movement density (log-likelihood)
        movement_loss = -1 * self.movement_density_block(movement_params, step_length, turn_angle, bearing)
        # print("movement_loss shape:", movement_loss.shape)

        # Combine losses with weights
        # total_loss = self.habitat_loss_weight * habitat_loss # habitat-only loss
        # total_loss = self.movement_loss_weight * movement_loss # movement-only loss
        total_loss = self.habitat_loss_weight * habitat_loss + self.movement_loss_weight * movement_loss

        # Check for NaNs in the combined predictions
        if torch.isnan(total_loss).any():
            print("NaNs detected in total_loss")
            print("total_lass:", total_loss)
            raise ValueError("NaNs detected in total_loss")

        # Apply the specified reduction
        if self.reduction == 'mean':
            return torch.mean(total_loss), torch.mean(habitat_loss), torch.mean(movement_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss), torch.sum(habitat_loss), torch.sum(movement_loss)
        elif self.reduction == 'none':
            return total_loss, habitat_loss, movement_loss

        # Default return (though it should never reach here without hitting an if)
        return total_loss, habitat_loss, movement_loss


