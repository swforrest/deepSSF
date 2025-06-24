"""
deepSSF_training_functions.py
==============

Description:
    This script contains functions for the deepSSF model training.
    - train_loop: The main training loop for the model.
    - test_loop: The main testing loop for the model.

Authors:
    Scott Forrest (scottwforrest@gmail.com)
    Dan Pagendam  (dan.pagendam@data61.csiro.au)

Date:
    2025-05-20
    
Usage:
    Run the script as a module, such as in a Jupyter notebook or Python console:
        >>> import deepSSF_training_functions
        >>> deepSSF_training_functions.train_loop(args)
    
    Or execute from the command line:
        $ python deepSSF_training_functions.py [optional arguments]
"""

# Standard library imports
import torch
import numpy as np

# Set the device to be used (GPU or CPU)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

### Training loop ###
"""
This code defines the main training loop for a single epoch. 
It iterates over batches from the training dataloader, 
moves the data to the correct device (e.g., CPU or GPU), calculates the loss, 
and performs backpropagation to update the model parameters. 
It also prints periodic updates of the current loss.
"""
def train_loop(dataloader_train, 
               model, 
               loss_fn, 
               optimisers, 
               skip_epoch0_training=False,
               batch_size=32):
    """
    Runs the training process for one epoch using the given dataloader, model,
    loss function, and optimizer. Prints progress updates every few batches.
    """

    # Unpack optimisers
    optimiser_movement, optimiser_habitat = optimisers

    # 1. Total number of training examples
    num_train_batches = len(dataloader_train)
    size = len(dataloader_train.dataset)

    # 2. Put model in training mode (affects layers like dropout, batchnorm)
    model.train()

    # 3. Variable to accumulate the total loss over the epoch
    epoch_loss = 0.0

    # 4. Loop over batches in the training dataloader
    for batch, (x1, x2, x3, y) in enumerate(dataloader_train):

        # Move the batch of data to the specified device (CPU/GPU)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = [item.to(device) for item in y]

        if isinstance(y, list):
            y = torch.stack(y)

        # Forward pass: compute the model output and loss
        with torch.set_grad_enabled(not skip_epoch0_training):
            outputs = model((x1, x2, x3))
            total_loss, habitat_loss, movement_loss = loss_fn(outputs, y)

        epoch_loss += total_loss.detach()  # Use detach to prevent memory leaks

        # Only perform optimization if not skipping training
        if not skip_epoch0_training:
            # Backpropagation: compute gradients and update parameters
            # Reset gradients before the next iteration

            # Zero all gradients
            optimiser_movement.zero_grad()
            optimiser_habitat.zero_grad()

            # Backward pass 
            # habitat_loss.backward(retain_graph=True)
            # movement_loss.backward()
            total_loss.backward()

            # For movement optimizer: save habitat gradients, then zero them out
            habitat_grads = []
            for param in model.conv_habitat.parameters():
                # Save the gradient
                if param.grad is not None:
                    habitat_grads.append(param.grad.clone())
                else:
                    habitat_grads.append(None)
                # Zero out habitat gradient for movement update
                param.grad = None

            # Update movement parameters
            optimiser_movement.step()

            # For habitat optimizer: restore habitat gradients and zero movement gradients
            for param in model.conv_movement.parameters():
                param.grad = None
            for param in model.fcn_movement_all.parameters():
                param.grad = None

            # Restore habitat gradients
            for i, param in enumerate(model.conv_habitat.parameters()):
                param.grad = habitat_grads[i]

            # Update habitat parameters
            optimiser_habitat.step()

        # Print an update every 5 batches to keep track of training progress
        if batch % 50 == 0:
            loss_val = total_loss.item()
            current = batch * batch_size + len(x1)
            if skip_epoch0_training:
                print(f"[Observation only] loss: {loss_val:>15f}  [{current:>5d}/{size:>5d}]")
            else:
                print(f"loss: {loss_val:>15f}  [{current:>5d}/{size:>5d}]")

        torch.cuda.empty_cache()

    # Compute the average training loss and print it
    epoch_loss /= num_train_batches
    if skip_epoch0_training:
        print(f"\nAvg training loss (observation only): {epoch_loss:>15f}")
    else:
        print(f"\nAvg training loss: {epoch_loss:>15f}")
    return epoch_loss
    # train_losses.append(epoch_loss.item())



## Test loop
"""
The test loop is similar to the training loop, 
but it does not perform backpropagation. 
It calculates the loss on the test set and returns the average loss.
"""
def test_loop(dataloader_test, model, loss_fn):
    """
    Evaluates the model on the provided test dataset by computing
    the average loss over all batches.
    No gradients are computed during this process (torch.no_grad()).
    """

    # 1. Set the model to evaluation mode (affects layers like dropout, batchnorm).
    model.eval()

    size = len(dataloader_test.dataset)
    num_batches = len(dataloader_test)

    test_loss = 0

    # 2. Disable gradient computation to speed up evaluation and reduce memory usage
    with torch.no_grad():
        # 3. Loop through each batch in the test dataloader
        for x1, x2, x3, y in dataloader_test:

            # Move the batch of data to the appropriate device (CPU/GPU)
            # x1, x2, x3 are the spatial covariates, temporal covariates, and bearing, respectively
            # y is the label (observed location of the next step)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device)

            if isinstance(y, list):
                y = torch.stack(y)

            # Compute the loss on the test set (no backward pass needed)
            total_loss, habitat_loss, movement_loss = loss_fn(model((x1, x2, x3)), y)
            test_loss += total_loss.detach()

    # 4. Compute average test loss over all batches
    test_loss /= num_batches

    torch.cuda.empty_cache()

    # Print the average test loss
    print(f"Avg test loss:    {test_loss:>15f} \n")
