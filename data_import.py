import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import pandas as pd

print(torch.rand(5,3,3))

# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# print(training_data)

# Print the length of the datasets
# print(f"Number of training examples: {len(training_data)}")
# print(f"Number of test examples: {len(test_data)}")

# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# ToTensor(training_data[0])
import rasterio

# Path to your TIF file
file_path = 'buffalo2005_ndvi_cent.tif'

# Step 1: Read the TIF file
# image = Image.open(file_path) # doesn't work as there are too many channels for the image file

# Using rasterio 
with rasterio.open(file_path) as ndvi:
    # Read all layers/channels into a single numpy array
    # rasterio indexes channels starting from 1, hence the range is 1 to src.count + 1
    ndvi_stack = ndvi.read([i for i in range(1, ndvi.count + 1)])

ndvi_stack
dir(ndvi_stack)

# using rasterio functions
# number of layers in the stack (rasterio object)
ndvi.count


for i in range(0, ndvi.count):
    plt.imshow(ndvi_stack[i])
    plt.show()
