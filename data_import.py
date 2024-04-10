import numpy as np
import matplotlib as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

print(torch.rand(5,3,3))

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(training_data)

# Print the length of the datasets
print(f"Number of training examples: {len(training_data)}")
print(f"Number of test examples: {len(test_data)}")

# Print a sample data point
sample = training_data[0]
print(f"Sample data point - features: {sample[0]}, target: {sample[1]}")


# Get the first sample from the training data
image = training_data[0], label = training_data[0]

# The image is a PyTorch tensor, so we need to convert it to a NumPy array to display it
image = image.numpy()

# The image has a shape of (1, 28, 28) because it's a grayscale image of size 28x28. 
# We need to remove the first dimension using np.squeeze to make it (28, 28) for displaying.
image = np.squeeze(image)

# Display the image
plt.imshow(image, cmap='gray')
plt.show()


# comment