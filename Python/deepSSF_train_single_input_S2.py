# %%
# ---
# title: "deepSSF Training - Single Layer Input"
# author:
#   - name: Scott Forrest
#     url: https://swforrest.github.io/
#     orcid: 0000-0001-9529-0108
#     affiliation: Queensland University of Technology, CSIRO
#     email: "scottwforrest@gmail.com"
# date: today
# format:
#     html:
#         toc: true
#         number_sections: true
#         code-fold: show
#         code-tools: true
#         code-overflow: scroll
#         # embed-resources: true
#         css: styles.css
# bibliography: references.bib
# abstract: |
#   In this script, we will train a deepSSF model on the training data. Instead of reading in the local
#   environmental layers from a tif file (raster stack), we will read in the local environmental layers
#   as individual images which are saved as numpy arrays (.npy). 
  
#   In this case the training data was generated using the `deepSSF_data_prep_single.ipynb` notebook, 
#   which crops out local images for each step of the observed telemetry data. The paths to the images
#   are referenced in the training data csv file.
# ---

# %% [markdown]
# # Detect computing environment
# 
# If using Google Colab, mount the drive and set the base directory to the working folder. If using local, set the base directory.

# %%
import os       # Operating system utilities
import sys

# Local environment setup
base_path = '..'
print("Running in local environment")

# Now you can use base_path regardless of environment
print(f"Using base path: {base_path}")


# %% [markdown]
# ## Import packages

# %%
print(sys.version)  # Print Python version in use

import numpy as np                                      # Array operations
import matplotlib.pyplot as plt                         # Plotting library
import torch                                            # Main PyTorch library
import torch.optim as optim                             # Optimization algorithms
import torch.nn as nn                                   # Neural network modules
import os                                               # Operating system utilities
import glob                                             # Pattern matching
import pandas as pd                                     # Data manipulation
import imageio.v2 as imageio                            # Image manipulation - for creating GIFs
import rasterio                                         # Raster data handling
import folium                                           # Interactive maps

from torch.utils.data import Dataset, DataLoader        # Dataset and batch data loading
from rasterio.plot import show                           # Plot raster data
from IPython.display import Image, display              # For plotting GIFs
from datetime import datetime, timedelta                # Date/time utilities
from tqdm import tqdm                                   # Progress bar
from pyproj import Transformer                          # Coordinate transformation

import deepSSF_model                                    # Import the .py file containing the deepSSF model     
import deepSSF_training_functions                       # Import the .py file containing the training functions
import deepSSF_loss_mixedLR as deepSSF_loss             # Import the .py file containing the deepSSF loss function
import deepSSF_early_stopping                           # Import the .py file containing the early stopping function  
import deepSSF_utils                                    # Import the .py file containing the utility functions 

# Get today's date
today_date = datetime.today().strftime('%Y-%m-%d')

# Set random seed for reproducibility
seed = 42

# %% [markdown]
# ## Set the device (accelerator - cuda for NVIDIA GPU or mps for Mac)

# %%
# Set the device to be used (GPU or CPU)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

if torch.backends.mps.is_available():
    # Set default tensor type for PyTorch
    torch.set_default_dtype(torch.float32)
    print('Set default tensor type to float32')

# %% [markdown]
# ### Create a directory to save the outputs
# 
# If we have already run this code today, we will add update index to create a new folder

# %%
window_size = 151  # Size of the input window

# Count existing directories with similar pattern
pattern = f'{base_path}/Python/outputs/model_training_S2/djelk_S2_nxn{window_size}_CNN_move_*_{today_date}'
existing_dirs = glob.glob(pattern)
dir_index = len(existing_dirs) + 1

# Create directory with index
output_dir = f'{base_path}/Python/outputs/model_training_S2/djelk_S2_nxn{window_size}_CNN_move_{dir_index}_{today_date}'
os.makedirs(output_dir, exist_ok=True)

print(f"Created directory: {output_dir}")

# To use an existing directory for loading trained model
# output_dir = f'{base_path}/Python/outputs/model_training_S2/id2005_2025-04-01'

# %% [markdown]
# # Import the data and set up the dataset and dataloader

# %%
class buffalo_data(Dataset):
    def __init__(self, csv_file, preload=True):
        self.data = csv_file
        self.npy_base_path = ''
        self.preload = preload
        
        # Process scalar columns as before
        scalar_columns = [
            'hour_t1_sin1', 
            'hour_t1_cos1', 
            'yday_t1_sin1', 
            'yday_t1_cos1',
        ]
        
        self.data[scalar_columns] = self.data[scalar_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.scalar_to_grid_data = torch.from_numpy(self.data[scalar_columns].values).float()
        
        # Process bearing as before
        self.data['bearing_tm1'] = pd.to_numeric(self.data['bearing'], errors='coerce').shift(1).fillna(0)
        self.bearing_tm1 = torch.from_numpy(self.data[['bearing_tm1']].values).float()
        
        # Preload all spatial data into memory if requested
        if self.preload:
            self.spatial_data_cache = []
            self.target_cache = []
            
            print("Preloading data into RAM...")
            for idx in tqdm(range(len(self.data))):

                # Get the paths for the .npy files
                s2_b1_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b1_path'])
                s2_b2_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b2_path'])
                s2_b3_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b3_path'])
                s2_b4_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b4_path'])
                s2_b5_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b5_path'])
                s2_b6_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b6_path'])
                s2_b7_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b7_path'])
                s2_b8_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b8_path'])
                s2_b8a_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b8a_path'])
                s2_b9_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b9_path'])
                s2_b11_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b11_path'])
                s2_b12_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['s2_b12_path'])
                # Slope
                slope_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['slope_path'])

                # Load the .npy files and convert to tensors
                # Sentinel-2 bands
                s2_b1_npy = np.load(s2_b1_path)
                s2_b2_npy = np.load(s2_b2_path)
                s2_b3_npy = np.load(s2_b3_path)
                s2_b4_npy = np.load(s2_b4_path)
                s2_b5_npy = np.load(s2_b5_path) 
                s2_b6_npy = np.load(s2_b6_path)
                s2_b7_npy = np.load(s2_b7_path)
                s2_b8_npy = np.load(s2_b8_path)
                s2_b8a_npy = np.load(s2_b8a_path)
                s2_b9_npy = np.load(s2_b9_path)
                s2_b11_npy = np.load(s2_b11_path)
                s2_b12_npy = np.load(s2_b12_path)
                # Slope
                slope_npy = np.load(slope_path)

                # Set NaN values to -1.0 for Sentinel-2 bands
                s2_b1_npy = np.nan_to_num(s2_b1_npy, nan=-1.0)
                s2_b2_npy = np.nan_to_num(s2_b2_npy, nan=-1.0)
                s2_b3_npy = np.nan_to_num(s2_b3_npy, nan=-1.0)
                s2_b4_npy = np.nan_to_num(s2_b4_npy, nan=-1.0)
                s2_b5_npy = np.nan_to_num(s2_b5_npy, nan=-1.0)
                s2_b6_npy = np.nan_to_num(s2_b6_npy, nan=-1.0)
                s2_b7_npy = np.nan_to_num(s2_b7_npy, nan=-1.0)
                s2_b8_npy = np.nan_to_num(s2_b8_npy, nan=-1.0)
                s2_b8a_npy = np.nan_to_num(s2_b8a_npy, nan=-1.0)
                s2_b9_npy = np.nan_to_num(s2_b9_npy, nan=-1.0)
                s2_b11_npy = np.nan_to_num(s2_b11_npy, nan=-1.0)
                s2_b12_npy = np.nan_to_num(s2_b12_npy, nan=-1.0)
                # Slope                
                slope_npy = np.nan_to_num(slope_npy, nan=0.0)

                # Convert to tensors
                s2_b1_tens = torch.tensor(s2_b1_npy, dtype=torch.float32)
                s2_b2_tens = torch.tensor(s2_b2_npy, dtype=torch.float32)
                s2_b3_tens = torch.tensor(s2_b3_npy, dtype=torch.float32)
                s2_b4_tens = torch.tensor(s2_b4_npy, dtype=torch.float32)
                s2_b5_tens = torch.tensor(s2_b5_npy, dtype=torch.float32)
                s2_b6_tens = torch.tensor(s2_b6_npy, dtype=torch.float32)
                s2_b7_tens = torch.tensor(s2_b7_npy, dtype=torch.float32)
                s2_b8_tens = torch.tensor(s2_b8_npy, dtype=torch.float32)
                s2_b8a_tens = torch.tensor(s2_b8a_npy, dtype=torch.float32)
                s2_b9_tens = torch.tensor(s2_b9_npy, dtype=torch.float32)
                s2_b11_tens = torch.tensor(s2_b11_npy, dtype=torch.float32)
                s2_b12_tens = torch.tensor(s2_b12_npy, dtype=torch.float32)
                # Slope                
                slope_tens = torch.tensor(slope_npy, dtype=torch.float32)
                
                spatial_data = torch.stack([s2_b1_tens, s2_b2_tens, s2_b3_tens, s2_b4_tens,
                                            s2_b5_tens, s2_b6_tens, s2_b7_tens, s2_b8_tens,
                                            s2_b8a_tens, s2_b9_tens, s2_b11_tens, s2_b12_tens, 
                                            slope_tens], dim=0).squeeze()
                
                # Append the spatial data tensor to the cache
                self.spatial_data_cache.append(spatial_data)
                
                # Load the target .tif file
                target_path = os.path.join(self.npy_base_path, self.data.iloc[idx]['target_path'])
                target = torch.tensor(np.load(target_path), dtype=torch.float32)

                # Append the target tensor to the cache
                self.target_cache.append(target)
            
            print("Data preloading complete!")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.preload:
            # Use preloaded data
            spatial_data_x = self.spatial_data_cache[index]
            target = self.target_cache[index]
        else:
            # Original disk-loading implementation
            # Get the paths for the .npy files
            s2_b1_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b1_path'])
            s2_b2_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b2_path'])
            s2_b3_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b3_path'])
            s2_b4_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b4_path'])
            s2_b5_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b5_path'])
            s2_b6_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b6_path'])
            s2_b7_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b7_path'])
            s2_b8_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b8_path'])
            s2_b8a_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b8a_path'])
            s2_b9_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b9_path'])
            s2_b11_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b11_path'])
            s2_b12_path = os.path.join(self.npy_base_path, self.data.iloc[index]['s2_b12_path'])
            slope_path = os.path.join(self.npy_base_path, self.data.iloc[index]['slope_path'])
                
            # Load the .npy files and convert to tensors
            # Sentinel-2 bands
            s2_b1_npy = np.load(s2_b1_path)
            s2_b2_npy = np.load(s2_b2_path)
            s2_b3_npy = np.load(s2_b3_path)
            s2_b4_npy = np.load(s2_b4_path)
            s2_b5_npy = np.load(s2_b5_path) 
            s2_b6_npy = np.load(s2_b6_path)
            s2_b7_npy = np.load(s2_b7_path)
            s2_b8_npy = np.load(s2_b8_path)
            s2_b8a_npy = np.load(s2_b8a_path)
            s2_b9_npy = np.load(s2_b9_path)
            s2_b11_npy = np.load(s2_b11_path)
            s2_b12_npy = np.load(s2_b12_path)

            # Set NaN values to -1.0 for Sentinel-2 bands
            s2_b1_npy = np.nan_to_num(s2_b1_npy, nan=-1.0)
            s2_b2_npy = np.nan_to_num(s2_b2_npy, nan=-1.0)
            s2_b3_npy = np.nan_to_num(s2_b3_npy, nan=-1.0)
            s2_b4_npy = np.nan_to_num(s2_b4_npy, nan=-1.0)
            s2_b5_npy = np.nan_to_num(s2_b5_npy, nan=-1.0)
            s2_b6_npy = np.nan_to_num(s2_b6_npy, nan=-1.0)
            s2_b7_npy = np.nan_to_num(s2_b7_npy, nan=-1.0)
            s2_b8_npy = np.nan_to_num(s2_b8_npy, nan=-1.0)
            s2_b8a_npy = np.nan_to_num(s2_b8a_npy, nan=-1.0)
            s2_b9_npy = np.nan_to_num(s2_b9_npy, nan=-1.0)
            s2_b11_npy = np.nan_to_num(s2_b11_npy, nan=-1.0)
            s2_b12_npy = np.nan_to_num(s2_b12_npy, nan=-1.0)

            # Convert to tensors
            s2_b1_tens = torch.tensor(s2_b1_npy, dtype=torch.float32)
            s2_b2_tens = torch.tensor(s2_b2_npy, dtype=torch.float32)
            s2_b3_tens = torch.tensor(s2_b3_npy, dtype=torch.float32)
            s2_b4_tens = torch.tensor(s2_b4_npy, dtype=torch.float32)
            s2_b5_tens = torch.tensor(s2_b5_npy, dtype=torch.float32)
            s2_b6_tens = torch.tensor(s2_b6_npy, dtype=torch.float32)
            s2_b7_tens = torch.tensor(s2_b7_npy, dtype=torch.float32)
            s2_b8_tens = torch.tensor(s2_b8_npy, dtype=torch.float32)
            s2_b8a_tens = torch.tensor(s2_b8a_npy, dtype=torch.float32)
            s2_b9_tens = torch.tensor(s2_b9_npy, dtype=torch.float32)
            s2_b11_tens = torch.tensor(s2_b11_npy, dtype=torch.float32)
            s2_b12_tens = torch.tensor(s2_b12_npy, dtype=torch.float32)
            
            # Slope
            slope_npy = np.load(slope_path)
            slope_npy = np.nan_to_num(slope_npy, nan=0.0)
            slope_tens = torch.tensor(slope_npy, dtype=torch.float32)
            
            spatial_data_x = torch.stack([s2_b1_tens, s2_b2_tens, s2_b3_tens, s2_b4_tens,
                                        s2_b5_tens, s2_b6_tens, s2_b7_tens, s2_b8_tens,
                                        s2_b8a_tens, s2_b9_tens, s2_b11_tens, s2_b12_tens, 
                                        slope_tens], dim=0).squeeze()
            
            # Load the target .tif file
            target_path = os.path.join(self.npy_base_path, self.data.iloc[index]['target_path'])
            target = torch.tensor(np.load(target_path), dtype=torch.float32)

        # Load the scalar values and bearing (these are already in memory)
        scalar_to_grid_data = self.scalar_to_grid_data[index]
        bearing_tm1 = self.bearing_tm1[index]

        return spatial_data_x, scalar_to_grid_data, bearing_tm1, target

# %%
# Load data into dataset
csv_file = f'/Users/scottforrest/deepSSF/nxn{window_size}/buffalo_djelk_nxn{window_size}_S2_all_steps_with_paths_n105763_steps.csv'
csv_data = pd.read_csv(csv_file)

# For testing, use a subset of the data
csv_data = csv_data[:100] # first rows
# csv_data = csv_data[csv_data['id'] == 2005] # first ID

csv_data.head()

# %%
# Create the dataset instance
dataset = buffalo_data(csv_data, preload=True)

training_split = 0.8
validation_split = 0.1
test_split = 0.1

dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset, [training_split, validation_split, test_split])
print(len(dataset_train))
print(len(dataset_val))
print(len(dataset_test))

# %% [markdown]
# ### Create dataloaders

# %%
os.cpu_count()

# %%
batch_size = 32 # batch size
num_workers = 0 # number of workers for data loader
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# %%
# Display image and label.
x1, x2, x3, labels = next(iter(dataloader_train))
print(f"Feature x1 batch shape: {x1.size()}")
print(f"Feature x2 batch shape: {x2.size()}")
print(f"Feature x3 batch shape: {x3.size()}")
print(f"Labels batch shape:     {labels.size()}")

# print(x3.detach().numpy())
print(x2[0,:])

# Plot the subset
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Convert the PyTorch tensor x2 to a NumPy array:
#   1) Detach from the computation graph so no gradients are tracked.
#   2) Move to CPU memory.
#   3) Convert to NumPy.
axs[0, 0].imshow(x1.detach().cpu().numpy()[0,0,:,:], cmap='viridis')
axs[0, 0].set_title('Local S2 Band 1')
axs[0, 1].imshow(x1.detach().cpu().numpy()[0,1,:,:], cmap='viridis')
axs[0, 1].set_title('Local S2 Band 2')
axs[1, 0].imshow(x1.detach().cpu().numpy()[0,2,:,:], cmap='viridis')
axs[1, 0].set_title('Local S2 Band 3')
axs[1, 1].imshow(x1.detach().cpu().numpy()[0,12,:,:], cmap='viridis')
axs[1, 1].set_title('Local Slope')

# Also plot the target as a single plot
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(labels.detach().numpy()[0,:,:], cmap='viridis')
ax.set_title('Target location (next step)')

# %% [markdown]
# # Load the model
# 
# As we have already described the model in detail in the `deepSSF_model` script, we can simply import the model here.
# 
# We will use the same model architecture as in the previous script, except that we will need to use a slightly edited dictionary to account for the additional input channels.

# %% [markdown]
# ## Define the parameters for the model
# 
# Here we enter the specific parameter values and hyperparameters for the model. 
# These are the values that will be used to instantiate the model.

# %%
n_max_pool_layers = 2 # used to determine the number of inputs entering the fully connected block - needs to be manually changed if the number of max pooling layers is changed
n_scalar_inputs = 4 # number of scalar inputs that are converted to a grid and appended to the spatial features

params_dict = {"batch_size": batch_size, #number of samples in each batch
               "image_dim": 151, #number of pixels along the edge of each local patch/image
               "pixel_size": 25, #number of metres along the edge of a pixel
               "input_channels": 13 + n_scalar_inputs, #number of spatial layers in each image + number of scalar layers that are converted to a grid
               "dim_in_nonspatial_to_grid": n_scalar_inputs, #the number of scalar predictors that are converted to a grid and appended to the spatial features
               "dense_dim_in_nonspatial": n_scalar_inputs, #change this to however many other scalar predictors you have (bearing, velocity etc)
               "kernel_size": 3, #the size of the 2D moving windows / kernels that are being learned
               "stride": 1, #the stride used when applying the kernel.  This reduces the dimension of the output if set to greater than 1
               "kernel_size_mp": 2, #the size of the kernel that is used in max pooling operations
               "stride_mp": 2, #the stride that is used in max pooling operations
               "padding": 1, #the amount of padding to apply to images prior to applying the 2D convolution
               "num_movement_params": 12, #number of parameters used to parameterise the movement kernel
               "dropout": 0.1, #the proportion of nodes that are dropped out in the dropout layers

               # hyperparameters that change the model architecture
               "output_channels": 4, #number of convolution filters to learn
               "output_channels_movement": 4, #number of convolution filters to learn for the movement kernel
               "dense_dim_hidden": 128, #number of nodes in the hidden layers

               # this will be updated below
               "dense_dim_in_all": 5476, #number of inputs entering the fully connected block once the nonspatial features have been concatenated to the spatial features
               "device": device
               }

# Now update the dictionary with calculated values
# params_dict["dense_dim_in_all"] = int(((params_dict["image_dim"] - (params_dict["image_dim"] % 2))**2) * (params_dict["output_channels_movement"] / (4**n_max_pool_layers)))

# %% [markdown]
# ## Instantiate the model
# 
# As described in the `deepSSF_train.ipynb` script, we saved the model definition into a file named `deepSSF_model.py`. We can instantiate the model by importing the file (which was done when importing other packages) and calling the classes parameter dictionary from that script.

# %%
params = deepSSF_model.ModelParams(params_dict)
model = deepSSF_model.ConvJointModel(params).to(device)
print(model)

# %% [markdown]
# # Pull out some testing data
# 
# To test the other blocks, and the full model, we will need some data. We can pull that out from the training set.

# %%
# Number of samples in the train dataset
print("Number of samples in the train dataset: ", len(dataloader_train.dataset))
print('\n')

# Select an index from the test dataset to retrieve a sample, between 0 and number of samples
# We picked this fairly arbitrarily, but with some interesting environmental features to illustrate the model's predictions
iteration_index = 70

# iteration_index = 1010
# iteration_index = 3971

# print(buffalo_df.iloc[iteration_index])

# return spatial_data_x, scalar_to_grid_data, bearing_tm1, target

# 2. Retrieve a single sample (features and label) from the test dataset at the specified index

# sample_spatial_covs is a sample of the spatial covariates for a single step
# sample_temporal_covs is a sample of the temporal covariates for a single step
# sample_prev_bearing is a sample bearing of the previous step
# sample_next_step is the target label (what we are trying to predict) for the next step

# We set these here and will also use them later in the script to check how the model's predictions look,
# and when we extract feature maps from the convolutional layers
sample_spatial_covs, sample_temporal_covs, sample_prev_bearing, sample_next_step = dataloader_train.dataset[iteration_index]

# 3. Reshape data tensors to add a batch dimension (since the model expects batches)
sample_spatial_covs = sample_spatial_covs.unsqueeze(0).to(device)
sample_temporal_covs = sample_temporal_covs.unsqueeze(0).to(device)
sample_prev_bearing = sample_prev_bearing.unsqueeze(0).to(device)
sample_next_step = sample_next_step.unsqueeze(0).to(device)

print(f'Shape of the sample spatial covariates:  {sample_spatial_covs.shape}')
print(f'Shape of the sample temporal covariates: {sample_temporal_covs.shape}')
print(f'Shape of the sample previous bearing:    {sample_prev_bearing.shape}')
print(f'Shape of the sample next step:           {sample_next_step.shape}')

# %%
# Plot the subset
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(sample_spatial_covs.detach().cpu().numpy()[0,0,:,:], cmap='viridis')
axs[0, 0].set_title('Local S2 Band 1')
axs[0, 1].imshow(sample_spatial_covs.detach().cpu().numpy()[0,1,:,:], cmap='viridis')
axs[0, 1].set_title('Local S2 Band 2')
axs[1, 0].imshow(sample_spatial_covs.detach().cpu().numpy()[0,2,:,:], cmap='viridis')
axs[1, 0].set_title('Local S2 Band 3')
axs[1, 1].imshow(sample_spatial_covs.detach().cpu().numpy()[0,12,:,:], cmap='viridis')
axs[1, 1].set_title('Local Slope')

# Also plot the target as a single plot
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(labels.detach().numpy()[0,:,:], cmap='viridis')
ax.set_title('Target location (next step)')
fig.savefig(f'{output_dir}/sample_covs_S2_bands.png', dpi=600)

# %% [markdown]
# ### Pull out the scalar values

# %%
# Extract the first sample (index 0) and its respective channel for each variable:
hour_t1_sin = sample_temporal_covs.detach().cpu().numpy()[0, 0]
hour_t1_cos = sample_temporal_covs.detach().cpu().numpy()[0, 1]
yday_t1_sin = sample_temporal_covs.detach().cpu().numpy()[0, 2]
yday_t1_cos = sample_temporal_covs.detach().cpu().numpy()[0, 3]

# Convert x3 similarly and extract the bearing from the first sample and channel:
bearing = sample_prev_bearing.detach().cpu().numpy()[0, 0]

hour_t1 = deepSSF_utils.recover_hour(hour_t1_sin, hour_t1_cos)
hour_t1_integer = int(hour_t1)  # Convert to integer
print(f'Hour:               {hour_t1_integer}')

yday_t1 = deepSSF_utils.recover_yday(yday_t1_sin, yday_t1_cos)
yday_t1_integer = int(yday_t1)  # Convert to integer
print(f'Day of the year:    {yday_t1_integer}')

bearing_degrees = np.degrees(bearing) % 360
bearing_degrees = round(bearing_degrees, 1)  # Round to 2 decimal places
bearing_degrees = int(bearing_degrees)  # Convert to integer
print(f'Bearing (radians):  {bearing}')
print(f'Bearing (degrees):  {bearing_degrees}')

# %% [markdown]
# ### Plot the sample covariates

# %%
# Plot the covariates
fig, axs = plt.subplots(2, 1, figsize=(5, 9))

red = sample_spatial_covs.detach().cpu()[0, 3, :, :]
green = sample_spatial_covs.detach().cpu()[0, 2, :, :]
blue = sample_spatial_covs.detach().cpu()[0, 1, :, :]

# Assuming b4_tens, b3_tens, and b2_tens are your tensors
rgb_image = torch.stack([red, green, blue], dim=-1)
print(rgb_image.shape)
# Convert to NumPy
rgb_image_np = rgb_image.numpy()

# Normalize to the range [0, 1] for display
rgb_image_np = (rgb_image_np - rgb_image_np.min()) / (rgb_image_np.max() - rgb_image_np.min())

# Plot RGB
im1 = axs[0].imshow(rgb_image_np)
axs[0].set_title('Sentinel-2 RGB Image')

# Plot Slope
im2 = axs[1].imshow(sample_spatial_covs.detach().cpu().numpy()[0,12,:,:], cmap='viridis')
axs[1].set_title('Slope')
# fig.colorbar(im2, ax=axs[1])

filename_covs = f'{output_dir}/covs_yday{yday_t1_integer}_hour{hour_t1_integer}.png'
plt.tight_layout()
plt.savefig(filename_covs, dpi=300, bbox_inches='tight') # if we want to save the figure
plt.show(block=False)  # Show the figure without blocking the execution
plt.close()  # Close the figure to free memory

# %% [markdown]
# ## Set model hyperparameters
# 
# Set the learning rate, loss function, optimizer, scheduler and early stopping. 

# %%
# learning_rate = 1e-3

# # Define the negative log-likelihood loss function with mean reduction
# loss_fn = deepSSF_loss.negativeLogLikeLoss(reduction='mean')

# # path to save the model weights
# path_save_weights = f'model_checkpoints/deepSSF_derived-covs_single_input_buffalo_{today_date}.pt'

# # Set up the Adam optimizer for updating the model's parameters
# optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# # Create a learning rate scheduler that reduces the LR by a factor of 0.1 
# #    if validation loss has not improved for 'patience=5' epochs
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimiser,  # The optimizer whose learning rate will be adjusted
#     mode='min', # The metric to be minimized (e.g., validation loss)
#     factor=0.1, # Factor by which the learning rate will be reduced
#     patience=5  # Number of epochs with no improvement before learning rate reduces
# )

# # EarlyStopping stops training after 'patience=10' epochs with no improvement, 
# #    optionally saving the best model weights
# early_stopping = deepSSF_early_stopping.EarlyStopping(patience=20, verbose=True, path=path_save_weights)

# %% [markdown]
# ## Training loop
# 
# This code defines the main training loop for a single epoch. It iterates over batches from the training dataloader, moves the data to the correct device (e.g., CPU or GPU), calculates the loss, and performs backpropagation to update the model parameters. It also prints periodic updates of the current loss.

# %%
train_loop = deepSSF_training_functions.train_loop
test_loop = deepSSF_training_functions.test_loop

# %% [markdown]
# ## Loss function

# %%
loss_fn = deepSSF_loss.negativeLogLikeLoss(reduction='mean')

# %% [markdown]
# ## Early stopping

# %%
early_stopping = deepSSF_early_stopping.EarlyStopping

# %% [markdown]
# # Train the model
# 
# Here we have the main training process that loops over multiple epochs. Each epoch involves:
# 
# 1. Training the model on a training dataset.
# 2. Validating the model on a validation dataset to monitor its performance and adjust the learning rate (via scheduler).
# 3. Checking for early stopping conditions. If triggered, the best model weights are restored, and a test evaluation is performed.
# 
# Additionally, commented-out code at the end shows how you might visualise and save intermediate training results (such as predicted probability surfaces) for diagnostic or research purposes. The saved images can then be combined into an animation.

# %%
print(f'Output directory: {output_dir}')
path_save_weights = f'{output_dir}/checkpoint_deepSSF_model.pt'
print(path_save_weights)

window_size = params_dict["image_dim"]  # Size of the input images

epochs = 15
train_losses = []  # Track training losses across epochs
val_losses = []   # Track validation losses across epochs
val_habitat_losses = []  # Track validation habitat losses across epochs
val_movement_losses = []  # Track validation movement losses across epochs
# Difference in loss between epochs
train_diff = []
val_diff = []
val_habitat_diff = []
val_movement_diff = []

# Initialize the parameter container using the parameters defined in 'params_dict'
params = deepSSF_model.ModelParams(params_dict)
# Create an instance of the ConvJointModel using the parameters,
# and move the model to the specified device (e.g., CPU or GPU)
model = deepSSF_model.ConvJointModel(params).to(device)
# Print the model architecture
print(model)

# Define the negative log-likelihood loss function with mean reduction
loss_fn = deepSSF_loss.negativeLogLikeLoss(reduction='mean') #, alpha=0.5

# Set the initial learning rates for each process
initial_learning_rate_movement = 1e-5
initial_learning_rate_habitat = 1e-4

# Create a combined optimiser for all movement-related parameters
movement_params = list(model.conv_movement.parameters()) + list(model.fcn_movement_all.parameters())
# movement_params = model.fcn_movement_all.parameters()

# Define separate optimizers for each component
optimiser_movement = optim.Adam(movement_params, lr=initial_learning_rate_movement)
optimiser_habitat = optim.Adam(model.conv_habitat.parameters(), lr=initial_learning_rate_habitat)

# Put optimisers into a tuple to call in the training loop
optimisers = (optimiser_movement, optimiser_habitat)

# Create separate schedulers for each optimizer
scheduler_movement = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser_movement, 'min', factor=0.1, patience=5)
scheduler_habitat = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser_habitat, 'min', factor=0.1, patience=5)

# Initialise early stopping 
early_stopping = deepSSF_early_stopping.EarlyStopping(patience=15, verbose=True, path=path_save_weights)

# Create directory for saving training images
os.makedirs(f'{output_dir}/training_images', exist_ok=True)
os.makedirs(f'{output_dir}/loss_images', exist_ok=True)

for t in range(epochs):

    # Initialise variables to store during training
    train_loss = 0.0
    num_train_batches = len(dataloader_train)

    val_loss = 0.0
    val_loss_habitat = 0.0
    val_loss_movement = 0.0
    num_batches = len(dataloader_val)

    print(f"Epoch {t+1}\n-------------------------------")

    # Skip training in the first epoch, but still calculate losses
    skip_training = (t == 0)

    # 1. Run the training loop for one epoch using the training dataloader
    epoch_loss = train_loop(dataloader_train, model, loss_fn, optimisers, 
                            skip_epoch0_training=skip_training,
                            batch_size=32)
    
    train_losses.append(epoch_loss.item())

    # 2. Evaluate model performance on the validation dataset
    model.eval()  # Switch to evaluation mode for proper layer behavior
    with torch.no_grad():

        # Loop through each batch in the validation dataloader
        for x1, x2, x3, y in dataloader_val:
            # Move data to the chosen device (CPU/GPU)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device)

            if isinstance(y, list):
                y = torch.stack(y)

            # Accumulate validation loss
            total_loss, habitat_loss, movement_loss = loss_fn(model((x1, x2, x3)), y)
            val_loss += total_loss.detach()
            val_loss_habitat += habitat_loss.detach()
            val_loss_movement += movement_loss.detach()

    # # 3. Step the scheduler based on the validation loss (adjusts learning rate if needed)
    # scheduler.step(val_loss)

    # Step the movement scheduler
    scheduler_movement.step(val_loss_movement)

    # Step the habitat scheduler regardless of the movement weight
    scheduler_habitat.step(val_loss_habitat)

    # 4. Compute the average validation loss and print it, along with the current learning rate
    val_loss /= num_batches
    val_loss_habitat /= num_batches
    val_loss_movement /= num_batches

    print(f"Avg validation loss:            {val_loss:>15f}")
    print(f"Avg validation habitat loss:    {val_loss_habitat:>15f}")
    print(f"Avg validation movement loss:   {val_loss_movement:>15f}")
    print(f"Movement learning rate:         {scheduler_movement.get_last_lr()}")
    print(f"Habitat learning rate:          {scheduler_habitat.get_last_lr()}")

    # 5. Track the validation loss for plotting or monitoring
    val_losses.append(val_loss.item())
    val_habitat_losses.append(val_loss_habitat.item())
    val_movement_losses.append(val_loss_movement.item())

    # Memory management - add after validation but before early stopping check
    # torch.mps.empty_cache()
    # gc.collect()

    # 6. Early stopping: if no improvement in validation loss for a set patience, stop training
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        # Restore the best model weights saved by EarlyStopping
        model.load_state_dict(torch.load(path_save_weights, weights_only=True, map_location=device))
        test_loop(dataloader_test, model, loss_fn)  # Evaluate on test set once training stops
        break
    else:
        model.eval()
        print("\n")

    torch.cuda.empty_cache()


    # ----------------------------------------------------
    # The following code demonstrates how
    # to optionally visualize or save intermediate results
    # (e.g., habitat probability surface, movement probability,
    # and next-step probability surfaces).

    # uncomment the code all in one go to run it (it should be inside the training loop)
    # ----------------------------------------------------

    # Extract training and validation losses for plotting

    # Convert the list of tensors to a single tensor
    train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
    val_losses_np = torch.tensor(val_losses).detach().cpu().numpy()
    val_habitat_losses_np = torch.tensor(val_habitat_losses).detach().cpu().numpy()
    val_movement_losses_np = torch.tensor(val_movement_losses).detach().cpu().numpy()

    # Get the difference in losses between epochs
    train_diff.append(train_losses_np[t] - train_losses_np[t-1])
    val_diff.append(val_losses_np[t] - val_losses_np[t-1])
    val_habitat_diff.append(val_habitat_losses_np[t] - val_habitat_losses_np[t-1])
    val_movement_diff.append(val_movement_losses_np[t] - val_movement_losses_np[t-1])

    # Number of epochs
    n_epochs = len(val_losses)

    # -----------------------------------------------------------
    # 1. Retrieve a single test example (covariates and labels)
    #    at the specified 'iteration_index' from the test dataset
    # -----------------------------------------------------------
    x1, x2, x3, labels = dataloader_train.dataset[int(iteration_index)]

    # -----------------------------------------------------------
    # 2. Add a batch dimension and move tensors to the device
    #    for model inference
    # -----------------------------------------------------------
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    x3 = x3.unsqueeze(0).to(device)

    # -----------------------------------------------------------
    # 3. Run the model on the single test example
    # -----------------------------------------------------------
    test = model((x1, x2, x3))

    # -----------------------------------------------------------
    # 4. Extract habitat and movement outputs;
    #    convert them to NumPy arrays for visualization
    # -----------------------------------------------------------
    hab_density = test.detach().cpu().numpy()[0, :, :, 0]
    movement_density = test.detach().cpu().numpy()[0, :, :, 1]

    # -----------------------------------------------------------
    # 5. Generate masks to exclude certain border cells for
    #    color scale reasons (setting them to -inf).
    # -----------------------------------------------------------
    x_mask = np.ones_like(hab_density)
    y_mask = np.ones_like(hab_density)

    # Mask out a few columns (0-2 and 98-end) and rows (0-2 and 98-end)
    x_mask[:, :3] = -np.inf
    x_mask[:, window_size-3:] = -np.inf
    y_mask[:3, :] = -np.inf
    y_mask[window_size-3:, :] = -np.inf

    # Apply the masks to habitat density
    hab_density_mask = hab_density * x_mask * y_mask

    # Combine habitat and movement densities to represent
    # next-step probability
    step_density = hab_density + movement_density
    step_density_mask = step_density * x_mask * y_mask

    # Plot the covariates
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5))

    # # Plot NDVI
    # im1 = axs[0, 0].imshow(ndvi_natural.numpy(), cmap='viridis')
    # axs[0, 0].set_title('NDVI')
    # fig.colorbar(im1, ax=axs[0, 0])

    # Plot Training and Validation Loss
    axs[0, 0].plot(range(n_epochs), train_losses_np, label='Training Loss', color='blue')
    axs[0, 0].plot(range(n_epochs), val_losses_np, label='Validation Loss', color='red')
    # axs[0, 0].plot(range(n_epochs), val_habitat_losses_np, label='Validation Habitat Loss', color='green')
    # axs[0, 0].plot(range(n_epochs), val_movement_losses_np, label='Validation Movement Loss', color='orange')
    axs[0, 0].set_xlim(0, epochs)
    axs[0, 0].set_title('Training and validation loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot habitat selection log-probability
    im2 = axs[0, 1].imshow(hab_density_mask, cmap='viridis')
    axs[0, 1].set_title('Habitat log-probability')
    fig.colorbar(im2, ax=axs[0, 1])

    # Plot movement log-probability
    im3 = axs[1, 0].imshow(movement_density, cmap='viridis')
    axs[1, 0].set_title('Movement log-probability')
    fig.colorbar(im3, ax=axs[1, 0])

    # Plot next-step log-probability
    im4 = axs[1, 1].imshow(step_density_mask, cmap='viridis')
    axs[1, 1].set_title('Next-step log-probability')
    fig.colorbar(im4, ax=axs[1, 1])

    filename_covs = f'{output_dir}/training_images/training_epoch_index{t}_yday{yday_t1_integer}_hour{hour_t1_integer}_bearing{bearing_degrees}.png'
    plt.tight_layout()
    plt.savefig(filename_covs, dpi=150) # creates inconsistent image sizes >>> , bbox_inches='tight'
    # plt.show(block=False)
    plt.close()  # Close the figure to free memory

    # Plot the difference in the loss of each component between epochs
    filename_diff = f'{output_dir}/loss_images/training_diff_epoch_index{t}_yday{yday_t1_integer}_hour{hour_t1_integer}_bearing{bearing_degrees}.png'
    plt.axhline(y=0, color='black', linestyle='--', label='Null Probability')  # null probs
    # plt.plot(range(n_epochs), train_diff, label='Training Loss Difference', color='blue')
    # plt.plot(range(n_epochs), val_diff, label='Validation Loss Difference', color='red')
    plt.plot(range(n_epochs), val_habitat_diff, label='Validation Habitat Loss Difference', color='green')
    plt.plot(range(n_epochs), val_movement_diff, label='Validation Movement Loss Difference', color='orange')
    plt.xlim(0, epochs)
    plt.title('Habitat and movement loss difference')
    plt.xlabel('Epoch')
    plt.ylabel('Loss difference')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(filename_diff, dpi=150) # creates inconsistent image sizes >>> , bbox_inches='tight'
    # plt.show(block=False)
    plt.close()  # Close the figure to free memory

print("Done!")

# %% [markdown]
# ### Make a GIF of the training images
# 
# First, here's a function to call to make a gif from a given directory.

# %%
# Example sorting by the epoch number
def extract_index(filename):
    # Extract the epoch number from the filename
    # Adjust the extraction based on your naming pattern
    import re
    match = re.search(r'index(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0

def create_gif(image_folder, output_filename, fps=10):
    """
    Creates a GIF from a sequence of images in a folder.

    Parameters:
    - image_folder: Path to the folder containing images
    - output_filename: Name of the output GIF file
    - fps: Frames per second for the GIF
    """
    import numpy as np
    from PIL import Image as PILImage
    
    # Get all png files in the specified folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=extract_index)

    # Check if any images were found
    if not images:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(images)} images")
    
    # Read all images and check their shapes
    frames = []
    shapes = []
    
    for i, image_path in enumerate(images):
        frame = imageio.imread(image_path)
        frames.append(frame)
        shapes.append(frame.shape)
        print(f"Image {i}: {os.path.basename(image_path)} - Shape: {frame.shape}")
    
    # Check if all shapes are the same
    if len(set(shapes)) > 1:
        print("Images have different shapes! Resizing to match the first image...")
        
        # Use the first image's shape as reference
        target_shape = shapes[0]
        target_height, target_width = target_shape[:2]
        
        # Resize all subsequent images to match
        resized_frames = []
        for i, frame in enumerate(frames):
            if frame.shape != target_shape:
                # Convert to PIL Image for resizing
                if len(frame.shape) == 3:  # Color image
                    pil_img = PILImage.fromarray(frame)
                else:  # Grayscale
                    pil_img = PILImage.fromarray(frame, mode='L')
                
                # Resize
                pil_img = pil_img.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                
                # Convert back to numpy array
                resized_frame = np.array(pil_img)
                resized_frames.append(resized_frame)
                print(f"Resized image {i} from {frame.shape} to {resized_frame.shape}")
            else:
                resized_frames.append(frame)
        
        frames = resized_frames
    else:
        print("All images have the same shape - no resizing needed")

    try:
        # Save as GIF
        imageio.mimsave(output_filename, frames, fps=fps, loop=0)
        
        # Display the GIF
        display(Image(filename=output_filename))
        
        print(f"GIF created successfully: {output_filename}")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        print("Frame shapes after processing:")
        for i, frame in enumerate(frames):
            print(f"  Frame {i}: {frame.shape}")


# %% [markdown]
# ## Create training GIF

# %%
# Path to your images
image_folder =  f'{output_dir}/training_images'
# Output GIF filename
output_filename = f'{output_dir}/training_gif_yday{yday_t1_integer}_hour{hour_t1_integer}_bearing{bearing_degrees}.gif'
# Create the GIF
create_gif(image_folder, output_filename, fps=10)

# %% [markdown]
# ## Create loss GIF

# %%
# Path to your images
image_folder =  f'{output_dir}/loss_images'
# Output GIF filename
output_filename = f'{output_dir}/loss_gif.gif'
# Create the GIF
create_gif(image_folder, output_filename, fps=10)

# %%
# to look at the parameters (weights and biases) of the model
# print(model.state_dict())

# %% [markdown]
# # Loading in previous models
# 
# As we've trained the model, the model parameters are already stored in the `model` object. But as we were training the model, we were saving it to file, and that, and other trained models can be loaded.
# 
# The model parameters that are being loaded must match the model object that has been defined above. If the model object has changed, the model parameters will not be able to be loaded.

# %%
path_save_weights

# %% [markdown]
# ### If loading a previously trained model

# %%
# to load previously saved weights
# path_save_weights = f'{output_dir}/checkpoint_deepSSF_buffalo2005_2025-04-01.pt'

model.load_state_dict(torch.load(path_save_weights,
                                 weights_only=True,
                                 map_location=torch.device('cpu')))

# %% [markdown]
# # View model outputs
# 
# ## Create a directory to save model outputs

# %% [markdown]
# ### Save the validation loss as a dataframe

# %%
# Directory for saving the loss dataframe
filename_loss_csv = f'{output_dir}/deepSSF_val_loss.csv'

# Check if val_losses is defined (which means a model has been trained in this session)
try:

    # Convert the list of tensors to a single tensor
    val_losses_tensor = torch.tensor(val_losses)

    print("val_losses has been defined - storing as csv\n")

    # Number of epochs
    n_epochs = len(val_losses)
    print(f'Number of epochs: {n_epochs}')

    val_losses_df = pd.DataFrame({
        "epoch": range(1, n_epochs + 1),
        "val_losses": val_losses_tensor.detach().cpu().numpy()
    })

    print(val_losses_df.head())

    # Save the validation losses to a CSV file
    val_losses_df.to_csv(filename_loss_csv, index=False)

# if val_losses hasn't been defined (for if you are loading model weights from a saved object)
except NameError:

    # This code runs if val_losses is not defined
    print("val_losses has not been defined - loading from saved csv\n")
    # Initialize it with a default value

    # Read the val_losses csv file
    val_losses_df = pd.read_csv(filename_loss_csv)
    print(val_losses_df.head())

    # Number of epochs
    n_epochs = len(val_losses_df)
    print(f'\nNumber of epochs: {n_epochs}')


# %%
# Directory for saving the loss dataframe
filename_train_loss_csv = f'{output_dir}/deepSSF_train_loss.csv'

# Check if train_losses is defined (which means a model has been trained in this session)
try:

    # Convert the list of tensors to a single tensor
    train_losses_tensor = torch.tensor(train_losses)

    print("train_losses has been defined - storing as csv\n")

    train_losses_df = pd.DataFrame({
        "epoch": np.linspace(1, n_epochs, len(train_losses)),
        "train_losses": train_losses_tensor.detach().cpu().numpy()
    })

    print(train_losses_df.head)

    # Save the train losses to a CSV file
    train_losses_df.to_csv(filename_train_loss_csv, index=False)

# if train_losses hasn't been defined (for if you are loading model weights from a saved object)
except NameError:

    # This code runs if train_losses is not defined
    print("train_losses has not been defined - loading from saved csv\n")
    # Initialize it with a default value

    # Read the train_losses csv file
    train_losses_df = pd.read_csv(filename_train_loss_csv)
    print(train_losses_df.head())


# %% [markdown]
# ### Plot the validation loss

# %%
# Directory for saving the loss plots
filename_loss = f'{output_dir}/val_loss.png'

# Plot the validation losses
plt.plot(train_losses_df['epoch'], train_losses_df['train_losses'], label='Training Loss', color='blue')  # Plot training loss in blue
plt.plot(val_losses_df['epoch'], val_losses_df['val_losses'], label='Validation Loss', color='red')  # Plot validation loss in red
plt.title('Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()  # Show legend to distinguish lines
plt.savefig(filename_loss, dpi=300, bbox_inches='tight')
plt.show(block=False)

# %% [markdown]
# # Test model
# 
# Take some random samples from the test dataset and generate predictions for them. We loop through the samples (which are shuffled randomly), make predictions, and plot the results.

# %%
# 1. Set the model in evaluation mode
model.eval()

# Loop over samples in the validation dataset
for i in range(0, 5):

  sample_number = np.random.randint(0, len(dataloader_test.dataset))
  print(f'Sample number: {sample_number}')

  # Display image and label
  x1, x2, x3, labels = dataloader_test.dataset[sample_number]

  # Add a batch dimension
  x1 = x1.unsqueeze(0).cpu()
  x2 = x2.unsqueeze(0).cpu()
  x3 = x3.unsqueeze(0).cpu()
  labels = labels.unsqueeze(0).cpu()

  # Pull out the scalars
  hour_t1_sin1 = x2.detach().numpy()[0,0]
  hour_t1_cos1 = x2.detach().numpy()[0,1]
  yday_t1_sin1 = x2.detach().numpy()[0,2]
  yday_t1_cos1 = x2.detach().numpy()[0,3]
  bearing = x3.detach().numpy()[0,0]

  # Recover the hour
  hour_t1 = deepSSF_utils.recover_hour(hour_t1_sin1, hour_t1_cos1)
  hour_t1_integer = int(hour_t1)  # Convert to integer
  print(f'Hour:                        {hour_t1_integer}')

  # Recover the day of the year
  yday_t1 = deepSSF_utils.recover_yday(yday_t1_sin1, yday_t1_cos1)
  yday_t1_integer = int(yday_t1)  # Convert to integer
  print(f'Day of the year:             {yday_t1_integer}')

  # Recover the bearing
  bearing_degrees = np.degrees(bearing) % 360
  bearing_degrees = round(bearing_degrees, 1)  # Round to 2 decimal places
  bearing_degrees = int(bearing_degrees)  # Convert to integer
  print(f'Bearing (radians):           {bearing}')
  print(f'Bearing (degrees):           {bearing_degrees}')

  # Pull out the RGB layers for plotting
  blue_layer = x1.detach().cpu().numpy()[0,1,:,:]
  green_layer = x1.detach().cpu().numpy()[0,2,:,:]
  red_layer = x1.detach().cpu().numpy()[0,3,:,:]

  # Stack the RGB layers
  rgb_image_np = np.stack([red_layer, green_layer, blue_layer], axis=-1)

  # Normalize to the range [0, 1] for display
  rgb_image_np = (rgb_image_np - rgb_image_np.min()) / (rgb_image_np.max() - rgb_image_np.min())

  # Find the coordinates of the element that is 1
  target = labels.detach().cpu().numpy()[0,:,:]
  coordinates = np.where(target == 1)

  # Extract the coordinates
  row, column = coordinates[0][0], coordinates[1][0]
  print(f"Next step is (row, column):  ({row}, {column})")


  # -------------------------------------------------------------------------
  # Run the model on the input data
  # -------------------------------------------------------------------------

  # Move input tensors to the GPU if available
  x1 = x1.to(device)
  x2 = x2.to(device)
  x3 = x3.to(device)

  test = model((x1, x2, x3))
  # print(test.shape)

  # Extract and exponentiate the habitat density channel
  hab_density = test.detach().cpu().numpy()[0, :, :, 0]
  hab_density_exp = np.exp(hab_density)
  # print(np.sum(hab_density_exp))  # Debug: check the sum of exponentiated values

  # Create masks to remove unwanted edge cells from visualization
  #    (setting them to -∞ affects the color scale in plots)
  x_mask = np.ones_like(hab_density)
  y_mask = np.ones_like(hab_density)

  # mask out cells on the edges that affect the colour scale
  x_mask[:, :3] = -np.inf
  x_mask[:, window_size-3:] = -np.inf
  y_mask[:3, :] = -np.inf
  y_mask[window_size-3:, :] = -np.inf

  # Apply the masks to the habitat density (log scale) and exponentiated version
  hab_density_mask = hab_density * x_mask * y_mask
  hab_density_exp_mask = hab_density_exp * x_mask * y_mask

  # Extract and exponentiate the movement density channel
  move_density = test.detach().cpu().numpy()[0,:,:,1]
  move_density_exp = np.exp(move_density)

  # Apply the same masking strategy to movement densities
  move_density_mask = move_density * x_mask * y_mask
  move_density_exp_mask = move_density_exp * x_mask * y_mask

  # Compute the next-step density by adding habitat + movement (log-space)
  step_density = test[0, :, :, 0] + test[0, :, :, 1]
  step_density = step_density.detach().cpu().numpy()
  step_density_exp = np.exp(step_density)

  # Apply masks to the step densities (log and exponentiated)
  step_density_mask = step_density * x_mask * y_mask
  step_density_exp_mask = step_density_exp * x_mask * y_mask

  # -------------------------------------------------------------------------
  # Plot the RGB image, slope, habitat selection, and movement density
  #   Change the panels to visualize different layers
  # -------------------------------------------------------------------------
  fig, axs = plt.subplots(2, 2, figsize=(10, 10))

  # Plot RGB
  im1 = axs[0, 0].imshow(rgb_image_np)
  axs[0, 0].set_title('Sentinel-2 RGB')

  # Plot slope
  im2 = axs[0, 1].imshow(x1.detach().cpu().numpy()[0,12,:,:], cmap='viridis')
  axs[0, 1].set_title('Slope')
  fig.colorbar(im2, ax=axs[0, 1], shrink=0.7)

  # Plot habitat selection
  im3 = axs[1, 0].imshow(hab_density_mask, cmap='viridis')
  axs[1, 0].set_title('Habitat selection log-probability')
  fig.colorbar(im3, ax=axs[1, 0], shrink=0.7)

  # # Movement density (change the axis and uncomment one of the other panels)
  # im3 = axs[1, 0].imshow(move_density_mask, cmap='viridis')
  # axs[1, 0].set_title('Movement log-probability')
  # fig.colorbar(im3, ax=axs[0, 1], shrink=0.7)

  # Next-step probability
  im4 = axs[1, 1].imshow(step_density_mask, cmap='viridis')
  axs[1, 1].set_title('Next-step log-probability')
  fig.colorbar(im4, ax=axs[1, 1], shrink=0.7)

  # Save the figure
  filename_covs = f'{output_dir}/deepSSF_S2_slope_id_yday{yday_t1_integer}_hour{hour_t1_integer}_bearing{bearing_degrees}_next_r{row}_c{column}.png'
  plt.tight_layout()
  plt.savefig(filename_covs, dpi=600, bbox_inches='tight')
  plt.show(block=False)
  plt.close()  # Close the figure to free memory


# %% [markdown]
# # Extracting convolution layer outputs
# 
# In the convolutional blocks, each convolutional layer learns a set of **filters** (kernels) that extract different features from the input data. In the habitat selection subnetwork, the convolution filters (and their associated bias parameters - not shown below) are the only parameters that are trained, and it is the filters that transform the set of input covariates into the habitat selection probabilities. They do this by maximising features of the inputs that correlate with observed next-steps.
# 
# For each convolutional layer, there are typically a number of filters. For the habitat selection subnetwork, we used 4 filters in the first two layers, and a single filter in the last layer. Each of these filters has a number of **channels** which correspond one-to-one with the input layers. The outputs of the filter channels are then combined to produce a feature map, with a single feature map produced for each filter. In successive layers, the feature maps become the input layers, and the filters operate on these layers. Because there are multiple filters in ech layer, they can 'specialise' in extracting different features from the input layers.
# 
# By visualizing and inspecting these filters, and the corresponding feature maps, we can:
# 
# - Gain interpretability: Understand what kind of features the network is detecting—e.g., edges, shapes, or textures.
# - Debug: Check if the filters have meaningful patterns or if something went wrong (e.g., all zeros or random noise).
# - Compare layers: See how early layers often learn low-level patterns while deeper layers learn more abstract features.
# 
# We will first set up some activation hooks for storing the feature maps. Activation hooks are placed at certain points within the model's forward pass and store intermediate results. We will also extract the convolution filters (which are weights of the model and as such don't require hooks - we can access them directly).
# 
# We will then run the sample covariates through the model and extract the feature maps from the habitat selection convolutional block, and plot them along with the covariates and convolution filters.
# 
# Note that there are also ReLU activation functions in the convolutional blocks, which are not shown below. These are applied to the feature maps, and set all negative values to zero. They are not learned parameters, but are part of the forward pass of the model.
# 

# %% [markdown]
# ### Create scalar grids for plotting
# 
# Using the `Scalar_to_Grid_Block` class from the `deepSSF_model` script, we can convert the scalar covariates into grids for plotting.

# %%
# Create an instance of the scalar-to-grid block using model parameters
scalar_to_grid_block = deepSSF_model.Scalar_to_Grid_Block(params)

# Convert scalars into spatial grid representation
scalar_maps = scalar_to_grid_block(x2)
print(scalar_maps.shape)  # Check the shape of the generated spatial maps

# %% [markdown]
# ## Convolutional layer 1
# 
# ### Activation hook

# %%
# -----------------------------------------------------------
# Create a dictionary to store activation outputs
# -----------------------------------------------------------
activation = {}

def get_activation(name):
    """
    Returns a hook function that can be registered on a layer
    to capture its output (i.e., feature maps) after the forward pass.

    Args:
        name (str): The key under which the activation is stored in the 'activation' dict.
    """
    def hook(model, input, output):
        # Detach and save the layer's output in the dictionary
        activation[name] = output.detach()
    return hook

# -----------------------------------------------------------
# Register a forward hook on the first convolution layer
#    in the model's 'conv_habitat' block
# -----------------------------------------------------------
model.conv_habitat.conv2d[0].register_forward_hook(get_activation("hab_conv1"))

# -----------------------------------------------------------
# Perform a forward pass through the model with the desired input
#    The feature maps from the hooked layer will be stored in 'activation'
# -----------------------------------------------------------
out = model((x1, x2, x3))  # e.g., model((spatial_data_x, scalars_to_grid, bearing_x))

# -----------------------------------------------------------
# Retrieve the captured feature maps from the dictionary
#    and move them to the CPU for inspection
# -----------------------------------------------------------
feat_maps1 = activation["hab_conv1"].cpu()
print("Feature map shape:", feat_maps1.shape)
# Typically shape: (batch_size, out_channels, height, width)

# -----------------------------------------------------------
# Visualize the feature maps for the first sample in the batch
# -----------------------------------------------------------
feat_maps1_sample = feat_maps1[0]  # Shape: (out_channels, H, W)
num_maps1 = feat_maps1_sample.shape[0]
print("Number of feature maps:", num_maps1)



# %% [markdown]
# ### Stack spatial and scalar (as grid) covariates
# 
# For plotting. Also create a vector of names to index over.

# %%
covariate_stack = torch.cat([x1, scalar_maps], dim=1)
print(covariate_stack.shape)

covariate_names = ['S2 B1',
                   'S2 B2',
                   'S2 B3',
                   'S2 B4',
                   'S2 B5',
                   'S2 B6',
                   'S2 B7',
                   'S2 B8',
                   'S2 B8a',
                   'S2 B9',
                   'S2 B11',
                   'S2 B12',
                   'Slope',
                   'Hour sin1',
                   'Hour cos1',
                   'Hour sin2',
                   'Hour cos2',
                   'yday sin1',
                   'yday cos1',
                   'yday sin2',
                   'yday cos2',]

# %% [markdown]
# ### Extract filters and plot

# %%
# -------------------------------------------------------------------------
# Check or print the convolution layer in conv_habitat (for debugging)
# -------------------------------------------------------------------------
print(model.conv_habitat.conv2d)

# -------------------------------------------------------------------------
# Set the model to evaluation mode (disables dropout, etc.)
# -------------------------------------------------------------------------
model.eval()

# -------------------------------------------------------------------------
# Extract the weights (filters) from the first convolution layer in conv_habitat
# -------------------------------------------------------------------------
filters_c1 = model.conv_habitat.conv2d[0].weight.data.clone().cpu()
print("Filters shape:", filters_c1.shape)
# Typically (out_channels, in_channels, kernel_height, kernel_width)

# -------------------------------------------------------------------------
# Visualize each filter’s first channel in a grid of subplots
# -------------------------------------------------------------------------
num_filters_c1 = filters_c1.shape[1]
print(num_filters_c1)

for z in range(num_maps1):

    fig, axes = plt.subplots(2, num_filters_c1, figsize=(2*num_filters_c1, 4))
    for i in range(num_filters_c1):

        # Add the covariates as the first row of subplots
        axes[0,i].imshow(covariate_stack[0, i].detach().cpu().numpy(), cmap='viridis')
        axes[0,i].axis('off')
        axes[0,i].set_title(f'{covariate_names[i]}')
        if i > x1.shape[1] - 1:
            im1 = axes[0,i].imshow(covariate_stack[0, i].detach().cpu().numpy(), cmap='viridis')
            im1.set_clim(-1, 1)
            axes[0,i].text(scalar_maps.shape[2] // 2, scalar_maps.shape[3] // 2,
                f'Value: {round(x2[0, i-x1.shape[1]].item(), 2)}',
                ha='center', va='center', color='white', fontsize=12)

        kernel = filters_c1[z, i, :, :]  # Show the first input channel
        im = axes[1,i].imshow(kernel, cmap='viridis')
        axes[1,i].axis('off')
        axes[1,i].set_title(f'Layer 1, Filter {z+1}')
        # Annotate each cell with the numeric value
        for (j, k), val in np.ndenumerate(kernel):
            axes[1,i].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/conv_layer1_filters{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)


    # -----------------------------------------------------------
    # Loop over each feature map channel and save them as images.
    #    Multiply by x_mask * y_mask if you need to mask out edges.
    # -----------------------------------------------------------

    plt.figure()
    plt.imshow(feat_maps1_sample[z].numpy() * x_mask * y_mask, cmap='viridis')
    plt.title(f"Layer 1, Feature Map {z+1}")
    # Hide axis if you prefer: plt.axis('off')
    plt.savefig(f'{output_dir}/conv_layer1_feature_map{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)



# %% [markdown]
# ## Convolutional layer 2
# 
# ### Activation hook

# %%
# -----------------------------------------------------------
# Register a forward hook on the second convolution layer
#    in the model's 'conv_habitat' block
# -----------------------------------------------------------
model.conv_habitat.conv2d[2].register_forward_hook(get_activation("hab_conv2"))

# -----------------------------------------------------------
# Perform a forward pass through the model with the desired input
#    The feature maps from the hooked layer will be stored in 'activation'
# -----------------------------------------------------------
out = model((x1, x2, x3))  # e.g., model((spatial_data_x, scalars_to_grid, bearing_x))

# -----------------------------------------------------------
# Retrieve the captured feature maps from the dictionary
#    and move them to the CPU for inspection
# -----------------------------------------------------------
feat_maps2 = activation["hab_conv2"].cpu()
print("Feature map shape:", feat_maps2.shape)
# Typically shape: (batch_size, out_channels, height, width)

# -----------------------------------------------------------
# Visualize the feature maps for the first sample in the batch
# -----------------------------------------------------------
feat_maps2_sample = feat_maps2[0]  # Shape: (out_channels, H, W)
num_maps2 = feat_maps2_sample.shape[0]
print("Number of feature maps:", num_maps2)



# %% [markdown]
# ### Extract filters and plot

# %%
# -------------------------------------------------------------------------
# Extract the weights (filters) from the second convolution layer in conv_habitat
# -------------------------------------------------------------------------
filters_c2 = model.conv_habitat.conv2d[2].weight.data.clone().cpu()
print("Filters shape:", filters_c2.shape)
# Typically (out_channels, in_channels, kernel_height, kernel_width)

# -------------------------------------------------------------------------
# Visualize each filter’s first channel in a grid of subplots
# -------------------------------------------------------------------------
num_filters_c2 = filters_c2.shape[1]
print(num_filters_c2)

for z in range(num_maps2):

    fig, axes = plt.subplots(2, num_filters_c2, figsize=(2*num_filters_c2, 4))
    for i in range(num_filters_c2):

        # Add the covariates as the first row of subplots
        axes[0,i].imshow(feat_maps1_sample[i].numpy() * x_mask * y_mask, cmap='viridis')
        axes[0,i].axis('off')
        axes[0,i].set_title(f"Layer 1, Map {z+1}")

        # if i > 3:
        #     im1 = axes[0,i].imshow(covariate_stack[0, i].detach().cpu().numpy(), cmap='viridis')
        #     im1.set_clim(-1, 1)
        #     axes[0,i].text(scalar_maps.shape[2] // 2, scalar_maps.shape[3] // 2,
        #         f'Value: {round(x2[0, i-4].item(), 2)}',
        #         ha='center', va='center', color='white', fontsize=12)

        kernel = filters_c2[z, i, :, :]  # Show the first input channel
        im = axes[1,i].imshow(kernel, cmap='viridis')
        axes[1,i].axis('off')
        axes[1,i].set_title(f'Layer 2, Filter {z+1}')
        # Annotate each cell with the numeric value
        for (j, k), val in np.ndenumerate(kernel):
            axes[1,i].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/conv_layer2_filters{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)


    # -----------------------------------------------------------
    # 6. Loop over each feature map channel and save them as images.
    #    Multiply by x_mask * y_mask if you need to mask out edges.
    # -----------------------------------------------------------

    plt.figure()
    plt.imshow(feat_maps2_sample[z].numpy() * x_mask * y_mask, cmap='viridis')
    plt.title(f"Layer 2, Feature Map {z+1}")
    # Hide axis if you prefer: plt.axis('off')
    plt.savefig(f'{output_dir}/conv_layer2_feature_map{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)



# %% [markdown]
# ## Convolutional layer 3
# 
# ### Activation hook

# %%
# -----------------------------------------------------------
# Register a forward hook on the third convolution layer
#    in the model's 'conv_habitat' block
# -----------------------------------------------------------
model.conv_habitat.conv2d[4].register_forward_hook(get_activation("hab_conv3"))

# -----------------------------------------------------------
# Perform a forward pass through the model with the desired input
#    The feature maps from the hooked layer will be stored in 'activation'
# -----------------------------------------------------------
out = model((x1, x2, x3))  # e.g., model((spatial_data_x, scalars_to_grid, bearing_x))

# -----------------------------------------------------------
# Retrieve the captured feature maps from the dictionary
#    and move them to the CPU for inspection
# -----------------------------------------------------------
feat_maps3 = activation["hab_conv3"].cpu()
print("Feature map shape:", feat_maps3.shape)
# Typically shape: (batch_size, out_channels, height, width)

# -----------------------------------------------------------
# Visualize the feature maps for the first sample in the batch
# -----------------------------------------------------------
feat_maps3_sample = feat_maps3[0]  # Shape: (out_channels, H, W)
num_maps3 = feat_maps3_sample.shape[0]
print("Number of feature maps:", num_maps3)



# %% [markdown]
# ### Extract filters and plot

# %%
# -------------------------------------------------------------------------
# Extract the weights (filters) from the second convolution layer in conv_habitat
# -------------------------------------------------------------------------
filters_c3 = model.conv_habitat.conv2d[4].weight.data.clone().cpu()
print("Filters shape:", filters_c3.shape)
# Typically (out_channels, in_channels, kernel_height, kernel_width)

# -------------------------------------------------------------------------
# Visualize each filter’s first channel in a grid of subplots
# -------------------------------------------------------------------------
num_filters_c3 = filters_c3.shape[1]
print(num_filters_c3)

for z in range(num_maps3):

    fig, axes = plt.subplots(2, num_filters_c3, figsize=(2*num_filters_c3, 4))
    for i in range(num_filters_c3):

        # Add the covariates as the first row of subplots
        axes[0,i].imshow(feat_maps2_sample[i].numpy() * x_mask * y_mask, cmap='viridis')
        axes[0,i].axis('off')
        axes[0,i].set_title(f"Layer 2, Map {z+1}")


        kernel = filters_c3[z, i, :, :]  # Show the first input channel
        im = axes[1,i].imshow(kernel, cmap='viridis')
        axes[1,i].axis('off')
        axes[1,i].set_title(f'Layer 3, Filter {z+1}')
        # Annotate each cell with the numeric value
        for (j, k), val in np.ndenumerate(kernel):
            axes[1,i].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/conv_layer3_filters{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)


    # -----------------------------------------------------------
    # 6. Loop over each feature map channel and save them as images.
    #    Multiply by x_mask * y_mask if you need to mask out edges.
    # -----------------------------------------------------------

    plt.figure()
    plt.imshow(feat_maps3_sample[z].numpy() * x_mask * y_mask, cmap='viridis')
    plt.title(f"Habitat selection log probability")
    # Hide axis if you prefer: plt.axis('off')
    plt.savefig(f'{output_dir}/conv_layer3_feature_map{z}_{today_date}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)



# %% [markdown]
# # Checking estimated movement parameters
# 
# Similarly to the convolutional layers, we can set hooks to extract the predicted movement parameters from the model, and assess how variable that is across samples.

# %%
# -------------------------------------------------------------------------
# Create a list to store the intermediate output from the fully connected
#    movement sub-network (fcn_movement_all)
# -------------------------------------------------------------------------
intermediate_output = []

def hook(module, input, output):
    """
    Hook function that captures the output of the specified layer
    (fcn_movement_all) during the forward pass.
    """
    intermediate_output.append(output)

# -------------------------------------------------------------------------
# Register the forward hook on 'fcn_movement_all', so its outputs
#    are recorded every time the model does a forward pass.
# -------------------------------------------------------------------------
hook_handle = model.fcn_movement_all.register_forward_hook(hook)

# -------------------------------------------------------------------------
# Perform a forward pass with the model in evaluation mode,
#    disabling gradient computation.
# -------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    final_output = model((x1, x2, x3))

# -------------------------------------------------------------------------
# Inspect the captured intermediate output
#    'intermediate_output[0]' corresponds to the first (and only) forward pass.
# -------------------------------------------------------------------------
print("Intermediate output shape:", intermediate_output[0].shape)
print("Intermediate output values:", intermediate_output[0][0])

# -------------------------------------------------------------------------
# Remove the hook to avoid repeated capturing in subsequent passes
# -------------------------------------------------------------------------
hook_handle.remove()

# -------------------------------------------------------------------------
# Unpack the parameters from the FCN output (assumes a specific ordering)
# -------------------------------------------------------------------------
gamma_shape1, gamma_scale1, gamma_weight1, \
gamma_shape2, gamma_scale2, gamma_weight2, \
vonmises_mu1, vonmises_kappa1, vonmises_weight1, \
vonmises_mu2, vonmises_kappa2, vonmises_weight2 = intermediate_output[0][0]

# -------------------------------------------------------------------------
# Convert parameters from log-space (if applicable) and print them
#    Gamma and von Mises parameters
# -------------------------------------------------------------------------
# --- Gamma #1 ---
print("Gamma shape 1:", torch.exp(gamma_shape1))
print("Gamma scale 1:", torch.exp(gamma_scale1))
print("Gamma weight 1:",
      torch.exp(gamma_weight1) / (torch.exp(gamma_weight1) + torch.exp(gamma_weight2)))

# --- Gamma #2 ---
print("Gamma shape 2:", torch.exp(gamma_shape2))
print("Gamma scale 2:", torch.exp(gamma_scale2) * 500)  # scale factor 500
print("Gamma weight 2:",
      torch.exp(gamma_weight2) / (torch.exp(gamma_weight1) + torch.exp(gamma_weight2)))

# --- von Mises #1 ---
# % (2*np.pi) ensures the mu (angle) is wrapped within [0, 2π)
print("Von Mises mu 1:", vonmises_mu1 % (2*np.pi))
print("Von Mises kappa 1:", torch.exp(vonmises_kappa1))
print("Von Mises weight 1:",
      torch.exp(vonmises_weight1) / (torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2)))

# --- von Mises #2 ---
print("Von Mises mu 2:", vonmises_mu2 % (2*np.pi))
print("Von Mises kappa 2:", torch.exp(vonmises_kappa2))
print("Von Mises weight 2:",
      torch.exp(vonmises_weight2) / (torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2)))


# %% [markdown]
# ## Plot the movement distributions
# 
# We can use the movement parameters to plot the step length and turning angle distributions for the sample covariates.

# %%
# -------------------------------------------------------------------------
# Define helper functions for calculating Gamma and von Mises log-densities
# -------------------------------------------------------------------------
def gamma_density(x, shape, scale):
    """
    Computes the log of the Gamma density for each value in x.

    Args:
      x (Tensor): Input values for which to compute the density.
      shape (float): Gamma shape parameter
      scale (float): Gamma scale parameter

    Returns:
      Tensor: The log of the Gamma probability density at each x.
    """
    return -1*torch.lgamma(shape) - shape*torch.log(scale) \
           + (shape - 1)*torch.log(x) - x/scale

def vonmises_density(x, kappa, vm_mu):
    """
    Computes the log of the von Mises density for each value in x.

    Args:
      x (Tensor): Input angles in radians.
      kappa (float): Concentration parameter (kappa)
      vm_mu (float): Mean direction parameter (mu)

    Returns:
      Tensor: The log of the von Mises probability density at each x.
    """
    return kappa*torch.cos(x - vm_mu) - 1*(np.log(2*torch.pi) + torch.log(torch.special.i0(kappa)))


# -------------------------------------------------------------------------
# Round and display the mixture weights for the Gamma distributions
# -------------------------------------------------------------------------
gamma_weight1_recovered = torch.exp(gamma_weight1)/(torch.exp(gamma_weight1) + torch.exp(gamma_weight2))
rounded_gamma_weight1 = round(gamma_weight1_recovered.item(), 2)

gamma_weight2_recovered = torch.exp(gamma_weight2)/(torch.exp(gamma_weight1) + torch.exp(gamma_weight2))
rounded_gamma_weight2 = round(gamma_weight2_recovered.item(), 2)

# -------------------------------------------------------------------------
# Round and display the mixture weights for the von Mises distributions
# -------------------------------------------------------------------------
vonmises_weight1_recovered = torch.exp(vonmises_weight1)/(torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2))
rounded_vm_weight1 = round(vonmises_weight1_recovered.item(), 2)

vonmises_weight2_recovered = torch.exp(vonmises_weight2)/(torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2))
rounded_vm_weight2 = round(vonmises_weight2_recovered.item(), 2)


# -------------------------------------------------------------------------
# 1. Plotting the Gamma mixture distribution
#    a) Generate x values
#    b) Compute individual Gamma log densities
#    c) Exponentiate and combine using recovered weights
# -------------------------------------------------------------------------
x_values = torch.linspace(1, 101, 1000).to(device)
gamma1_density = gamma_density(x_values, torch.exp(gamma_shape1), torch.exp(gamma_scale1))
gamma2_density = gamma_density(x_values, torch.exp(gamma_shape2), torch.exp(gamma_scale2)*500)
gamma_mixture_density = gamma_weight1_recovered*torch.exp(gamma1_density) \
                        + gamma_weight2_recovered*torch.exp(gamma2_density)

# Move results to CPU and convert to NumPy for plotting
x_values_np = x_values.cpu().numpy()
gamma1_density_np = np.exp(gamma1_density.cpu().numpy())
gamma2_density_np = np.exp(gamma2_density.cpu().numpy())
gamma_mixture_density_np = gamma_mixture_density.cpu().numpy()

# -------------------------------------------------------------------------
# 2. Plot the Gamma distributions and their mixture
# -------------------------------------------------------------------------
plt.plot(x_values_np, gamma1_density_np, label=f'Gamma 1 Density: weight = {rounded_gamma_weight1}')
plt.plot(x_values_np, gamma2_density_np, label=f'Gamma 2 Density: weight = {rounded_gamma_weight2}')
plt.plot(x_values_np, gamma_mixture_density_np, label='Gamma Mixture Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gamma Density Function')
plt.legend()
plt.show(block=False)


# -------------------------------------------------------------------------
# 3. Plotting the von Mises mixture distribution
#    a) Generate x values from -π to π
#    b) Compute individual von Mises log densities
#    c) Exponentiate and combine using recovered weights
# -------------------------------------------------------------------------
x_values = torch.linspace(-np.pi, np.pi, 1000).to(device)
vonmises1_density = vonmises_density(x_values, torch.exp(vonmises_kappa1), vonmises_mu1)
vonmises2_density = vonmises_density(x_values, torch.exp(vonmises_kappa2), vonmises_mu2)
vonmises_mixture_density = vonmises_weight1_recovered*torch.exp(vonmises1_density) \
                           + vonmises_weight2_recovered*torch.exp(vonmises2_density)

# Move results to CPU and convert to NumPy for plotting
x_values_np = x_values.cpu().numpy()
vonmises1_density_np = np.exp(vonmises1_density.cpu().numpy())
vonmises2_density_np = np.exp(vonmises2_density.cpu().numpy())
vonmises_mixture_density_np = vonmises_mixture_density.cpu().numpy()

# -------------------------------------------------------------------------
# 4. Plot the von Mises distributions and their mixture
# -------------------------------------------------------------------------
plt.plot(x_values_np, vonmises1_density_np, label=f'Von Mises 1 Density: weight = {rounded_vm_weight1}')
plt.plot(x_values_np, vonmises2_density_np, label=f'Von Mises 2 Density: weight = {rounded_vm_weight2}')
plt.plot(x_values_np, vonmises_mixture_density_np, label='Von Mises Mixture Density')
plt.xlabel('x (radians)')
plt.ylabel('Density')
plt.title('Von Mises Density Function')
plt.ylim(0, 0.4)  # Set a limit for the y-axis
plt.legend()
plt.show(block=False)


# %% [markdown]
# ## Generate a distribution of movement parameters
# 
# To see how variable the movement parameters are across samples, we can generate a distribution of movement parameters from a batch of samples.
# 
# We take the code from above that we used to create the DataLoader for the test data and increase the batch size (to get more samples to create the distribution from).
# 
# As we're not using the test dataset any more, we'll just put all of the samples in the same batch, and generate movement parameters for all of them.

# %%
print(f'There are {len(dataset_test)} samples in the test dataset')
bs = len(dataset_test) # batch size
dataloader_test = DataLoader(dataset=dataset_test, batch_size=bs, shuffle=True)

# %% [markdown]
# Take all of the samples from the test dataset and put them in a single batch.

# %%
# -----------------------------------------------------------
# Fetch a batch of data from the training dataloader
# -----------------------------------------------------------
x1_batch, x2_batch, x3_batch, labels = next(iter(dataloader_test))

x1_batch = x1_batch.to(device)
x2_batch = x2_batch.to(device)
x3_batch = x3_batch.to(device)
labels = labels.to(device)

# -----------------------------------------------------------
# Register a forward hook to capture the outputs
#    from 'fcn_movement_all' during the forward pass
# -----------------------------------------------------------
hook_handle = model.fcn_movement_all.register_forward_hook(hook)

# -----------------------------------------------------------
# Perform a forward pass in evaluation mode to generate
#    and capture the sub-network's outputs in 'intermediate_output'
# -----------------------------------------------------------
model.eval()  # Disables certain layers like dropout

# Pass the batch through the model
final_output = model((x1_batch, x2_batch, x3_batch))

# -----------------------------------------------------------
# Prepare lists to store the distribution parameters
#    for each sample in the batch
# -----------------------------------------------------------
gamma_shape1_list = []
gamma_scale1_list = []
gamma_weight1_list = []
gamma_shape2_list = []
gamma_scale2_list = []
gamma_weight2_list = []
vonmises_mu1_list = []
vonmises_kappa1_list = []
vonmises_weight1_list = []
vonmises_mu2_list = []
vonmises_kappa2_list = []
vonmises_weight2_list = []

# -----------------------------------------------------------
# Extract parameters from 'intermediate_output'
#    for every sample in the batch
# -----------------------------------------------------------
for batch_output in intermediate_output:
    # Each 'batch_output' corresponds to one forward pass;
    # it might contain multiple samples if the batch size > 1
    for sample_output in batch_output:
        # Unpack the 12 parameters of the Gamma and von Mises mixtures
        gamma_shape1, gamma_scale1, gamma_weight1, \
        gamma_shape2, gamma_scale2, gamma_weight2, \
        vonmises_mu1, vonmises_kappa1, vonmises_weight1, \
        vonmises_mu2, vonmises_kappa2, vonmises_weight2 = sample_output

        # Convert log-space parameters to real space, then store
        gamma_shape1_list.append(torch.exp(gamma_shape1).item())
        gamma_scale1_list.append(torch.exp(gamma_scale1).item())
        gamma_weight1_list.append(
            (torch.exp(gamma_weight1)/(torch.exp(gamma_weight1) + torch.exp(gamma_weight2))).item()
        )
        gamma_shape2_list.append(torch.exp(gamma_shape2).item())
        gamma_scale2_list.append((torch.exp(gamma_scale2)*500).item())  # scale factor 500
        gamma_weight2_list.append(
            (torch.exp(gamma_weight2)/(torch.exp(gamma_weight1) + torch.exp(gamma_weight2))).item()
        )
        vonmises_mu1_list.append((vonmises_mu1 % (2*np.pi)).item())
        vonmises_kappa1_list.append(torch.exp(vonmises_kappa1).item())
        vonmises_weight1_list.append(
            (torch.exp(vonmises_weight1)/(torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2))).item()
        )
        vonmises_mu2_list.append((vonmises_mu2 % (2*np.pi)).item())
        vonmises_kappa2_list.append(torch.exp(vonmises_kappa2).item())
        vonmises_weight2_list.append(
            (torch.exp(vonmises_weight2)/(torch.exp(vonmises_weight1) + torch.exp(vonmises_weight2))).item()
        )


# %% [markdown]
# ### Plot the distribution of movement parameters

# %%
# -----------------------------------------------------------
# Define a helper function to plot histograms
#    for the collected parameters
# -----------------------------------------------------------
def plot_histogram(data, title, xlabel):
    """
    Plots a histogram of the provided data.

    Args:
        data (list): Data points to plot in a histogram.
        title (str): Title of the histogram plot.
        xlabel (str): X-axis label.
    """
    plt.figure()
    plt.hist(data, bins=30, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show(block=False)

# -----------------------------------------------------------
# Plot histograms for each parameter distribution
# -----------------------------------------------------------
plot_histogram(gamma_shape1_list, 'Gamma Shape 1 Distribution', 'Shape 1')
plot_histogram(gamma_scale1_list, 'Gamma Scale 1 Distribution', 'Scale 1')
plot_histogram(gamma_weight1_list, 'Gamma Weight 1 Distribution', 'Weight 1')
plot_histogram(gamma_shape2_list, 'Gamma Shape 2 Distribution', 'Shape 2')
plot_histogram(gamma_scale2_list, 'Gamma Scale 2 Distribution', 'Scale 2')
plot_histogram(gamma_weight2_list, 'Gamma Weight 2 Distribution', 'Weight 2')
plot_histogram(vonmises_mu1_list, 'Von Mises Mu 1 Distribution', 'Mu 1')
plot_histogram(vonmises_kappa1_list, 'Von Mises Kappa 1 Distribution', 'Kappa 1')
plot_histogram(vonmises_weight1_list, 'Von Mises Weight 1 Distribution', 'Weight 1')
plot_histogram(vonmises_mu2_list, 'Von Mises Mu 2 Distribution', 'Mu 2')
plot_histogram(vonmises_kappa2_list, 'Von Mises Kappa 2 Distribution', 'Kappa 2')
plot_histogram(vonmises_weight2_list, 'Von Mises Weight 2 Distribution', 'Weight 2')

# -----------------------------------------------------------
# Remove the hook to stop capturing outputs
#    in subsequent forward passes
# -----------------------------------------------------------
hook_handle.remove()

# %% [markdown]
# # Importing spatial data
# 
# Instead of importing the stacks of local layers (one for each step), here we want to import the spatial covariates for the extent we want to simulate over. We use an extent that covers all of the observed locations, which refer to as the 'landscape'.

# %% [markdown]
# ## Sentinel-2 bands
# 
# Each stack represents a month of median values of cloud-free pixels, and each layer in the stack are the bands.
# 
# During the data preparation all of these layers were scaled by 10,000, and don't need to be scaled any further.

# %%
# Specify the directory containing your TIFF files
data_dir = f'{base_path}/mapping/cropped rasters/sentinel2/25m'  # Replace with the actual path to your TIFF files

# Use glob to get a list of all TIFF files matching the pattern
tif_files = glob.glob(os.path.join(data_dir, 'S2_SR_masked_scaled_25m_*.tif'))
print(f'Found {len(tif_files)} TIFF files')
print('\n'.join(tif_files))

# %%
# Initialise a dictionary to store data with date as the key
data_dict = {}

# Loop over each TIFF file to read and process the data
for tif_file in tif_files:
    # Extract the filename from the path
    filename = os.path.basename(tif_file)

    # Extract the date from the filename
    # Assuming filenames are in the format 'S2_SR_masked_YYYY_MM.tif'
    date_str = filename.replace('S2_SR_masked_scaled_25m_', '').replace('.tif', '')
    # date_str will be something like '2019_01'

    # Read the TIFF file using rasterio
    with rasterio.open(tif_file) as src:
        # Read all bands of the TIFF file
        data = src.read()
        # 'data' is a NumPy array with shape (bands, height, width)

        # Count the number of cells that are NaN
        n_nan = np.isnan(data).sum()

        print(f"Date: {date_str}")
        print(f"Number of NaN values in {date_str}: {n_nan}")
        print(f'Proportion of NaN values: {n_nan / data.size:.4%}\n')

        # Replace NaN values with zeros
        data = np.nan_to_num(data, nan=0)

        # Add the data to the dictionary with date as the key
        data_dict[date_str] = data


# %%
# Select some bands from the processed data stored in 'data_dict' for plotting
layers_to_plot = []

# Specify the date and band numbers you want to plot
dates_to_plot = ['2019_01', '2019_05']  # This grabs all available dates. You can select specific ones if needed.
bands_to_plot = [1, 2, 3]  # Band indices for bands 2, 3, and 4, which are B, G, and R

# Loop through the selected dates and bands to prepare them for plotting
for date_str in dates_to_plot:
    data = data_dict[date_str]  # Get the normalized data for this date

    for band_idx in bands_to_plot:
        # Collect the specific band for plotting
        layers_to_plot.append((data[band_idx], band_idx + 1, date_str))

# Plot the stored layers
for band, band_number, date_str in layers_to_plot:
    plt.figure(figsize=(8, 6))
    plt.imshow(band, cmap='viridis')
    plt.title(f'Band {band_number} - {date_str}')
    plt.colorbar() #label='Normalized Value'
    plt.show(block=False)


# %% [markdown]
# ### Plot as RGB
# 
# We can also visualise the Sentinel-2 bands as an RGB image, using the Red, Green and Blue bands.
# 
# The plotting was a bit dark so we will adjust the brightness of the image using a gamma correction.

# %%
# Specify the date for the RGB layers
date_str = '2019_08'

# pull out the RGB bands
r_band = data_dict[date_str][3]
g_band = data_dict[date_str][2]
b_band = data_dict[date_str][1]

# Stack the bands along a new axis
rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

# Apply gamma correction to the image
gamma = 1.75
rgb_image = rgb_image ** (1/gamma)

plt.figure()  # Create a new figure
plt.imshow(rgb_image)
plt.title('Sentinel 2 RGB')
plt.show(block=False)
plt.close()  # Close the figure to free memory

# %% [markdown]
# ## Slope

# %%
# Path to the slope raster file
file_path = f'{base_path}/mapping/cropped rasters/slope.tif'

# read the raster file
with rasterio.open(file_path) as src:
    # Read the raster band as separate variable
    slope_landscape = src.read(1)
    # Get the metadata of the raster
    slope_meta = src.meta
    raster_transform = src.transform # same as the raster transform in the NDVI raster read

# %%
# Check the slope metadata:
print("Slope metadata:")
print(slope_meta)
print("\n")

# Check the shape (rows, columns) of the slope landscape raster:
print("Shape of slope landscape raster:")
print(slope_landscape.shape)
print("\n")

# Check for NA values in the slope raster:
print("Number of NA values in the slope raster:")
print(np.isnan(slope_landscape).sum())

# Replace NaNs in the slope array with 0.0 (representing water):
slope_landscape = np.nan_to_num(slope_landscape, nan=0.0)

# Define the maximum and minimum slope values from the stack of local layers:
slope_max = 12.2981
slope_min = 0.0006

# Convert the slope landscape data from a NumPy array to a PyTorch tensor:
slope_landscape_tens = torch.from_numpy(slope_landscape)

# Normalize the slope landscape data:
slope_landscape_norm = (slope_landscape_tens - slope_min) / (slope_max - slope_min)

# Visualize the slope landscape (note: displaying the original tensor, not the normalised data):
plt.imshow(slope_landscape_tens.numpy())
plt.colorbar()
plt.title('Slope (natural scale)')
plt.show(block=False)


# %% [markdown]
# ## Subset function
# 
# As we described the subset function in the `deepSSF_simulations.ipynb` notebook, and stored it in the `deepSSF_functions.py` script, we will just import it here.

# %%
subset_function = deepSSF_utils.subset_raster_with_padding_torch

# %% [markdown]
# ### Testing the subset function
# 
# We want to ensure that the function pads the raster when it is outside the landscape extent.

# %%
# Pick a location (x, y) from the buffalo DataFrame
x = csv_data['x1_'].iloc[0]
y = csv_data['y1_'].iloc[0]

# Get the subset of the slope landscape
slope_subset, origin_x, origin_y = subset_function(slope_landscape_norm, x, y, window_size, raster_transform)

# For sentinel 2 data
selected_month = '2019_01'
# Get the data for the selected month
s2_data = data_dict[selected_month]

# Convert the NumPy array to a PyTorch tensor
s2_tensor = torch.from_numpy(s2_data)
s2_tensor = s2_tensor.float()  # Ensure the tensor is of type float
print(s2_tensor.shape) # [bands, height, width]

# Get the subset of the Sentinel-2 bands
s2_b1_subset, origin_x, origin_y = subset_function(s2_tensor[0,:,:], x, y, window_size, raster_transform)
s2_b2_subset, origin_x, origin_y = subset_function(s2_tensor[1,:,:], x, y, window_size, raster_transform)
s2_b3_subset, origin_x, origin_y = subset_function(s2_tensor[2,:,:], x, y, window_size, raster_transform)
s2_b4_subset, origin_x, origin_y = subset_function(s2_tensor[3,:,:], x, y, window_size, raster_transform)
s2_b5_subset, origin_x, origin_y = subset_function(s2_tensor[4,:,:], x, y, window_size, raster_transform)
s2_b6_subset, origin_x, origin_y = subset_function(s2_tensor[5,:,:], x, y, window_size, raster_transform)
s2_b7_subset, origin_x, origin_y = subset_function(s2_tensor[6,:,:], x, y, window_size, raster_transform)
s2_b8_subset, origin_x, origin_y = subset_function(s2_tensor[7,:,:], x, y, window_size, raster_transform)
s2_b8a_subset, origin_x, origin_y = subset_function(s2_tensor[8,:,:], x, y, window_size, raster_transform)
s2_b9_subset, origin_x, origin_y = subset_function(s2_tensor[9,:,:], x, y, window_size, raster_transform)
s2_b11_subset, origin_x, origin_y = subset_function(s2_tensor[10,:,:], x, y, window_size, raster_transform)
s2_b12_subset, origin_x, origin_y = subset_function(s2_tensor[11,:,:], x, y, window_size, raster_transform)

# Plot the subset
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(s2_b2_subset.detach().numpy(), cmap='viridis')
axs[0, 0].set_title('Band 2 (blue) Subset')

axs[0, 1].imshow(s2_b3_subset.detach().numpy(), cmap='viridis')
axs[0, 1].set_title('Band 3 (green) Subset')

axs[1, 0].imshow(s2_b4_subset.detach().numpy(), cmap='viridis')
axs[1, 0].set_title('Band 4 (red) Subset')

axs[1, 1].imshow(slope_subset.detach().numpy(), cmap='viridis')
axs[1, 1].set_title('Slope Subset')


# %% [markdown]
# ## Create a mask for edge cells
# 
# Due to the padding at the edges of the covariates, convolutional layers create artifacts that can affect the colour scale of the predictions when plotting. To avoid this, we will create a mask that we can apply to the predictions to remove the edge cells.

# %%
# Create a mask to remove the edge values for plotting
# (as it affects the colour scale)
x_mask = np.ones_like(slope_subset)
y_mask = np.ones_like(slope_subset)

# Mask out bordering cells
x_mask[:, :3] = -np.inf
x_mask[:, window_size-3:] = -np.inf
y_mask[:3, :] = -np.inf
y_mask[window_size-3:, :] = -np.inf

# %% [markdown]
# # Setup validation
# 
# To get the validation running we need a few extra functions.
# 
# Firstly, we need to index the Sentinel-2 layers correctly, based on the time of the simulated location. We'll do this by creating a function that takes day of the year of the simulated location and returns the correct index for the Sentinel-2 layers.
# 
# This indexing is slightly different from the indexing we used for the `deepSSF_simulations.ipynb` notebook, which was indexing NDVI layers. In that case we were indexing the layers directly, and therefore the first entry was at 0 (i.e., March was in month_index = 2). Here, we are creating a string that corresponds to the layer name, and therefore the first entry is at 1. (i.e., March will be at month_index = 3)

# %%
# Create a mapping from day of the year to month index
def day_to_month_index(day_of_year):
    # Calculate the year and the day within that year
    base_date = datetime(2019, 1, 1)
    date = base_date + timedelta(days=int(day_of_year) - 1)
    year_diff = date.year - base_date.year
    month_index = (date.month) + (year_diff * 12)  # month index (1-based)
    if month_index == 0:
        month_index += 1
    return month_index

yday = 35
month_index = day_to_month_index(yday)
print(month_index)

# %% [markdown]
# ### Check the Sentinel-2 layer indexing
# 
# Subset the raster layers at the first observed location of the training data.

# %%
# Step index for the buffalo data
step_index = 15

# starting location of buffalo 2005
x = csv_data['x1_'].iloc[step_index]
y = csv_data['y1_'].iloc[step_index]
print(f'Starting x and y coordinates: {x}, {y}')

yday = csv_data['yday_t1'].iloc[step_index]
print(f'Starting day of the year:     {yday}')

# Get the month index from the day of the year
month_index = day_to_month_index(yday)

# for sentinel 2 data
selected_month = f'2019_{month_index:02d}'
# Get the normalized data for the selected month
s2_data = data_dict[selected_month]

# Convert the NumPy array to a PyTorch tensor
s2_tensor = torch.from_numpy(s2_data)
s2_tensor = s2_tensor.float()  # Ensure the tensor is of type float
print(s2_tensor.shape)

# Get the subset of the Sentinel-2 bands
s2_b1_subset, origin_x, origin_y = subset_function(s2_tensor[0,:,:], x, y, window_size, raster_transform)
s2_b2_subset, origin_x, origin_y = subset_function(s2_tensor[1,:,:], x, y, window_size, raster_transform)
s2_b3_subset, origin_x, origin_y = subset_function(s2_tensor[2,:,:], x, y, window_size, raster_transform)
s2_b4_subset, origin_x, origin_y = subset_function(s2_tensor[3,:,:], x, y, window_size, raster_transform)
s2_b5_subset, origin_x, origin_y = subset_function(s2_tensor[4,:,:], x, y, window_size, raster_transform)
s2_b6_subset, origin_x, origin_y = subset_function(s2_tensor[5,:,:], x, y, window_size, raster_transform)
s2_b7_subset, origin_x, origin_y = subset_function(s2_tensor[6,:,:], x, y, window_size, raster_transform)
s2_b8_subset, origin_x, origin_y = subset_function(s2_tensor[7,:,:], x, y, window_size, raster_transform)
s2_b8a_subset, origin_x, origin_y = subset_function(s2_tensor[8,:,:], x, y, window_size, raster_transform)
s2_b9_subset, origin_x, origin_y = subset_function(s2_tensor[9,:,:], x, y, window_size, raster_transform)
s2_b11_subset, origin_x, origin_y = subset_function(s2_tensor[10,:,:], x, y, window_size, raster_transform)
s2_b12_subset, origin_x, origin_y = subset_function(s2_tensor[11,:,:], x, y, window_size, raster_transform)

# Get the subset of the slope landscape
slope_subset, origin_x, origin_y = subset_function(slope_landscape_norm, x, y, window_size, raster_transform)

# Plot the subset
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(s2_b2_subset.numpy(), cmap='viridis')
axs[0, 0].set_title('Band 2 (blue) Subset')

axs[0, 1].imshow(s2_b3_subset.numpy(), cmap='viridis')
axs[0, 1].set_title('Band 3 (green) Subset')

axs[1, 0].imshow(s2_b4_subset.numpy(), cmap='viridis')
axs[1, 0].set_title('Band 4 (red) Subset')

axs[1, 1].imshow(slope_subset.numpy(), cmap='viridis')
axs[1, 1].set_title('Slope Subset')

# %% [markdown]
# ### Plot as RGB

# %%
# pull out the RGB bands
r_band = s2_b4_subset.detach().numpy()
g_band = s2_b3_subset.detach().numpy()
b_band = s2_b2_subset.detach().numpy()

# Stack the bands along a new axis
rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

plt.figure()  # Create a new figure
plt.imshow(rgb_image)
plt.title('Sentinel 2 RGB')
plt.show(block=False)
plt.close()  # Close the figure to free memory

# %% [markdown]
# # Next-step probability values
# 
# We can now calculate the next-step probabilities for each observed step. As we generate habitat selection, movement and next-step probability surfaces, we can get the predicted probability values for each one, which can be compared to the respective process in the SSF.
# 
# The process for generating the next-step probabilities is as follows:
# 
# 1. Get the current location of the individual
# 2. Crop out the local layers for the current location
# 3. Run the model of the local layers to get the habitat selection, movement and next-step probability surfaces
# 4. Get the predicted probability values at the location of the next step
# 5. Store the predicted probability values and export them as a csv for comparison with the SSF
# 
# First, select the data to generate prediction values for. For testing the function we can select a subset.

# %%
# To select a subset of samples to test the function
# test_data = buffalo_df.iloc[0:10]

buffalo_id = 2005  # ID of the buffalo to test

# To select all of the data
test_data = csv_data[csv_data['id'] == buffalo_id] # first ID

# Get the number of samples in the test data
n_samples = len(test_data)
print(f'Number of samples: {n_samples}')

# Create empty vectors to store the predicted probabilities
habitat_probs = np.repeat(0., n_samples)
move_probs = np.repeat(0., n_samples)
next_step_probs = np.repeat(0., n_samples)

# %% [markdown]
# ## Loop over each step

# %%
# Create directory for saving prediction images
os.makedirs(f'{output_dir}/prediction_images', exist_ok=True)

# Start at 1 so the bearing at t - 1 is available
for i in range(1, n_samples):

  sample = test_data.iloc[i]

  # Current location (x1, y1)
  x = sample['x1_']
  y = sample['y1_']

  # Convert geographic coordinates to pixel coordinates
  px, py = ~raster_transform * (x, y)

  # Next step location (x2, y2)
  x2 = sample['x2_']
  y2 = sample['y2_']

  # Convert geographic coordinates to pixel coordinates
  px2, py2 = ~raster_transform * (x2, y2)

  # The difference in x and y coordinates
  d_x = x2 - x
  d_y = y2 - y
  # print('d_x and d_y are ', d_x, d_y) # Debugging

  # Temporal covariates for t1
  hour_t1_sin1 = sample['hour_t1_sin1']
  hour_t1_cos1 = sample['hour_t1_cos1']
  yday_t1_sin1 = sample['yday_t1_sin1']
  yday_t1_cos1 = sample['yday_t1_cos1']

  # Bearing of previous step (t - 1)
  bearing = sample['bearing_tm1']

  # Hour of the day (for saving the plot)
  hour_t1 = sample['hour_t1']

  # Day of the year
  yday = sample['yday_t1']

  # Convert day of the year to month index
  month_index = day_to_month_index(yday)
  # print(month_index)

  # For sentinel 2 data
  selected_month = f'2019_{month_index:02d}'
  # Get the Sentinel-2 layers for the selected month
  s2_data = data_dict[selected_month]

  # Convert the Sentinel-2 data from a NumPy array to a PyTorch tensor
  s2_tensor = torch.from_numpy(s2_data)
  s2_tensor = s2_tensor.float()  # Ensure the tensor is of type float
  # print(s2_tensor.shape)

  # Crop out the Sentinel-2 subsets at the location of x1, y1
  s2_b1_subset, origin_x, origin_y = subset_function(s2_tensor[0,:,:], x, y, window_size, raster_transform)
  s2_b2_subset, origin_x, origin_y = subset_function(s2_tensor[1,:,:], x, y, window_size, raster_transform)
  s2_b3_subset, origin_x, origin_y = subset_function(s2_tensor[2,:,:], x, y, window_size, raster_transform)
  s2_b4_subset, origin_x, origin_y = subset_function(s2_tensor[3,:,:], x, y, window_size, raster_transform)
  s2_b5_subset, origin_x, origin_y = subset_function(s2_tensor[4,:,:], x, y, window_size, raster_transform)
  s2_b6_subset, origin_x, origin_y = subset_function(s2_tensor[5,:,:], x, y, window_size, raster_transform)
  s2_b7_subset, origin_x, origin_y = subset_function(s2_tensor[6,:,:], x, y, window_size, raster_transform)
  s2_b8_subset, origin_x, origin_y = subset_function(s2_tensor[7,:,:], x, y, window_size, raster_transform)
  s2_b8a_subset, origin_x, origin_y = subset_function(s2_tensor[8,:,:], x, y, window_size, raster_transform)
  s2_b9_subset, origin_x, origin_y = subset_function(s2_tensor[9,:,:], x, y, window_size, raster_transform)
  s2_b11_subset, origin_x, origin_y = subset_function(s2_tensor[10,:,:], x, y, window_size, raster_transform)
  s2_b12_subset, origin_x, origin_y = subset_function(s2_tensor[11,:,:], x, y, window_size, raster_transform)

  # Crop out the slope subset at the location of x1, y1
  slope_subset, origin_x, origin_y = subset_function(slope_landscape_norm, x, y, window_size, raster_transform)

  # Location of the next step in local pixel coordinates
  px2_subset = px2 - origin_x
  py2_subset = py2 - origin_y
  # print('px2_subset and py2_subset are ', px2_subset, py2_subset) # Debugging

  # Stack the channels along a new axis
  x1 = torch.stack([s2_b1_subset,
                    s2_b2_subset,
                    s2_b3_subset,
                    s2_b4_subset,
                    s2_b5_subset,
                    s2_b6_subset,
                    s2_b7_subset,
                    s2_b8_subset,
                    s2_b8a_subset,
                    s2_b9_subset,
                    s2_b11_subset,
                    s2_b12_subset,
                    slope_subset], dim=0)

  # Add a batch dimension (required to be the correct dimension for the model)
  x1 = x1.unsqueeze(0).to(device)
  # print(x1.shape)

  # Temporal covariates for t1
  hour_t1_sin1_tensor = torch.tensor(hour_t1_sin1).float()
  hour_t1_cos1_tensor = torch.tensor(hour_t1_cos1).float()
  yday_t1_sin1_tensor = torch.tensor(yday_t1_sin1).float()
  yday_t1_cos1_tensor = torch.tensor(yday_t1_cos1).float()

  # Stack tensors
  x2 = torch.stack((hour_t1_sin1_tensor.unsqueeze(0),
                    hour_t1_cos1_tensor.unsqueeze(0),
                    yday_t1_sin1_tensor.unsqueeze(0),
                    yday_t1_cos1_tensor.unsqueeze(0)),
                    dim=1).to(device)
  # print(x2)
  # print(x2.shape)

  # put bearing in the correct dimension (batch_size, 1)
  bearing = torch.tensor(bearing).float().unsqueeze(0).unsqueeze(0).to(device)
  # print(bearing)
  # print(bearing.shape)

  # -------------------------------------------------------------------------
  # Run the model
  # -------------------------------------------------------------------------
  model_output = model((x1, x2, bearing))


  # -------------------------------------------------------------------------
  # Habitat selection probability
  # -------------------------------------------------------------------------
  hab_density = model_output.detach().cpu().numpy()[0,:,:,0]
  hab_density_exp = np.exp(hab_density)

  # Normalise the probability surface to sum to 1
  hab_density_exp_norm = hab_density_exp / np.sum(hab_density_exp)
  # print(np.sum(hab_density_exp_norm))  # Should be 1

  # Store the probability of habitat selection at the location of x2, y2
  # These probabilities are normalised in the model function
  habitat_probs[i] = hab_density_exp_norm[(int(py2_subset), int(px2_subset))]
  # print('Habitat probability value = ', habitat_probs[i])


  # -------------------------------------------------------------------------
  # Movement probability
  # -------------------------------------------------------------------------
  move_density = model_output.detach().cpu().numpy()[0,:,:,1]
  move_density_exp = np.exp(move_density)

  # Normalise the probability surface to sum to 1
  move_density_exp_norm = move_density_exp / np.sum(move_density_exp)
  # print(np.sum(move_density_exp_norm))  # Should be 1

  # Store the movement probability at the location of x2, y2
  # These probabilities are normalised in the model function
  move_probs[i] = move_density_exp_norm[(int(py2_subset), int(px2_subset))]
  # print('Movement probability value = ', move_probs[i])


  # -------------------------------------------------------------------------
  # Next step probability
  # -------------------------------------------------------------------------
  step_density = hab_density + move_density
  step_density_exp = np.exp(step_density)
  # print('Sum of step density exp = ', np.sum(step_density_exp)) # Won't be 1

  step_density_exp_norm = step_density_exp / np.sum(step_density_exp)
  # print('Sum of step density exp norm = ', np.sum(step_density_exp_norm)) # Should be 1

  # Extract the value of the covariates at the location of x2, y2
  next_step_probs[i] = step_density_exp_norm[(int(py2_subset), int(px2_subset))]
  # print('Next-step probability value = ', next_step_probs[i])


  # -------------------------------------------------------------------------
  # Plot the next-step predictions
  # -------------------------------------------------------------------------

  # Plot the first few probability surfaces - change the condition to i < n_steps to plot all
  if i < 51:

    # Mask out bordering cells
    hab_density_mask = hab_density * x_mask * y_mask
    move_density_mask = move_density * x_mask * y_mask
    step_density_mask = step_density * x_mask * y_mask

    # Create a mask for the next step
    next_step_mask = np.ones_like(hab_density)
    next_step_mask[int(py2_subset), int(px2_subset)] = -np.inf

    # Plot the outputs
    fig_out, axs_out = plt.subplots(2, 2, figsize=(10, 8))

    # RGB for plotting
    # pull out the RGB bands
    r_band = s2_b4_subset.detach().numpy()
    g_band = s2_b3_subset.detach().numpy()
    b_band = s2_b2_subset.detach().numpy()

    # Stack the bands along a new axis
    rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
    # Normalize to the range [0, 1] for display
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    # Plot s2_b2
    im1 = axs_out[0, 0].imshow(rgb_image)
    axs_out[0, 0].set_title('Sentinel 2 RGB')

    # Plot habitat selection log-probability
    im2 = axs_out[0, 1].imshow(hab_density_mask * next_step_mask, cmap='viridis')
    axs_out[0, 1].set_title('Habitat selection log-probability')
    fig_out.colorbar(im2, ax=axs_out[0, 1], shrink=0.7)

    # Movement density log-probability
    im3 = axs_out[1, 0].imshow(move_density_mask * next_step_mask, cmap='viridis')
    axs_out[1, 0].set_title('Movement log-probability')
    fig_out.colorbar(im3, ax=axs_out[1, 0], shrink=0.7)

    # Next-step probability
    im4 = axs_out[1, 1].imshow(step_density_mask * next_step_mask, cmap='viridis')
    axs_out[1, 1].set_title('Next-step log-probability')
    fig_out.colorbar(im4, ax=axs_out[1, 1], shrink=0.7)

    filename_covs = f'{output_dir}/prediction_images/id{buffalo_id}_step_index{i+1}_yday{yday}_hour{hour_t1}.png'
    plt.tight_layout()
    plt.savefig(filename_covs, dpi=150) #, bbox_inches='tight'
    # plt.show(block=False)
    plt.close()  # Close the figure to free memory

# %%
print(next_step_probs)

# %% [markdown]
# ### Make a GIF of the prediction images

# %%
# Path to your images
image_folder =  f'{output_dir}/prediction_images'
# Output GIF filename
output_filename = f'{output_dir}/prediction_gif_id{buffalo_id}_yday{yday_t1_integer}_hour{hour_t1_integer}_bearing{bearing_degrees}_next_r{row}_c{column}.gif'
# Create the GIF
create_gif(image_folder, output_filename, fps=5)

# %% [markdown]
# ## Calculate the null probabilities
# 
# As each cell has a probability values, we can calculate what the probability would be if the model provided no information at all, and each cell was equally likely to be the next step. This is just 1 divided by the total number of cells.

# %%
null_prob = 1 / (window_size ** 2)
print(f'Null probability: {null_prob:.3e}')

# %% [markdown]
# ## Compute the rolling average of the probabilities

# %%
rolling_window_size = 100 # Rolling window size

# Convert to pandas Series and compute rolling mean
rolling_mean_habitat = pd.Series(habitat_probs).rolling(window=window_size, center=True).mean()
rolling_mean_movement = pd.Series(move_probs).rolling(window=window_size, center=True).mean()
rolling_mean_next_step = pd.Series(next_step_probs).rolling(window=window_size, center=True).mean()

# %% [markdown]
# # Plot the probabilities
# 
# We can get an idea of how variable the probabilities are for the habitat selection and movement surfaces, and for the next-step probabilities, by plotting them across the trajectory

# %%
# Plot the habitat probs through time as a line graph
plt.plot(habitat_probs[range(100)], color='blue', label='Habitat Probabilities - S2')
plt.plot(rolling_mean_habitat[range(100)], color='red', label='Rolling Mean')
plt.axhline(y=null_prob, color='black', linestyle='--', label='Null Probability')  # null probs
plt.xlabel('Index')
plt.ylabel('Probability')
plt.title('Habitat Probability')
plt.legend()  # Add legend to differentiate lines
plt.show(block=False)
plt.savefig(f'{output_dir}/id{buffalo_id}_habitat_probs_100_steps.png', dpi=300, bbox_inches='tight')

# Plot the habitat probs through time as a line graph
plt.plot(habitat_probs[habitat_probs > 0], color='blue', label='Habitat Probabilities - S2')
plt.plot(rolling_mean_habitat[rolling_mean_habitat > 0], color='red', label='Rolling Mean')
plt.axhline(y=null_prob, color='black', linestyle='--', label='Null Probability')  # null probs
plt.xlabel('Index')
plt.ylabel('Probability')
plt.ylim(0, 5e-4)  # Set a limit for the y-axis
plt.title('Habitat Probability')
plt.legend()  # Add legend to differentiate lines
plt.show(block=False)
plt.savefig(f'{output_dir}/id{buffalo_id}_habitat_probs.png', dpi=300, bbox_inches='tight')

# Plot the movement probs through time as a line graph
plt.plot(move_probs[move_probs > 0], color='blue', label='Movement Probabilities - S2')
plt.plot(rolling_mean_movement[rolling_mean_movement > 0], color='red', label='Rolling Mean')
plt.axhline(y=null_prob, color='black', linestyle='--', label='Null Probability')  # null probs
plt.xlabel('Index')
plt.ylabel('Probability')
plt.title('Movement Probability')
plt.legend()  # Add legend to differentiate lines
plt.show(block=False)
plt.savefig(f'{output_dir}/id{buffalo_id}_move_probs.png', dpi=300, bbox_inches='tight')

# Plot the next step probs through time as a line graph
plt.plot(next_step_probs[next_step_probs > 0], color='blue', label='Next Step Probabilities - S2')
plt.plot(rolling_mean_next_step[rolling_mean_next_step > 0], color='red', label='Rolling Mean')
plt.axhline(y=null_prob, color='black', linestyle='--', label='Null Probability')  # null probs
plt.xlabel('Index')
plt.ylabel('Probability')
plt.title('Next Step Probability')
plt.legend()  # Add legend to differentiate lines
plt.show(block=False)
plt.savefig(f'{output_dir}/id{buffalo_id}_next_step_probs.png', dpi=300, bbox_inches='tight')

# %% [markdown]
# # Save the probabilities
# 
# We can save the probabilities to a csv file to compare with the SSF probabilities.

# %%
# Append the probabilities to the dataframe
test_data['habitat_probs'] = habitat_probs
test_data['move_probs'] = move_probs
test_data['next_step_probs'] = next_step_probs

csv_filename = f'{output_dir}/deepSSF_validation_id{buffalo_id}_n{len(test_data)}.csv'
print(csv_filename)
test_data.to_csv(csv_filename, index=True)

# %% [markdown]
# # Landscape habitat selection predictions

# %% [markdown]
# ## Select a smaller extent of the landscape
# 
# To illustrate the approach, we will select a smaller extent of the landscape to predict over, which covers the spatial extent of the training data. 

# %%
buffalo_df = csv_data[csv_data['id'] == buffalo_id] # first ID

# from the buffalo data
buffer = 1250
min_x = min(buffalo_df['x1_']) - buffer
max_x = max(buffalo_df['x1_']) + buffer
min_y = min(buffalo_df['y1_']) - buffer
max_y = max(buffalo_df['y1_']) + buffer

# custom extent in epsg:3112
# min_x = 28148.969145
# max_x = 47719.496935
# min_y = -1442210.335861
# max_y = -1433133.681746

# Convert geographic coordinates to pixel coordinates
min_px, min_py = ~raster_transform * (min_x, min_y)
print(min_px, min_py)
max_px, max_py = ~raster_transform * (max_x, max_y)
print(max_px, max_py)

# Round pixel coordinates to integers
min_px, max_px, min_py, max_py = int(round(min_px)), \
    int(round(max_px)), \
        int(round(min_py)), \
            int(round(max_py))

# Print the pixel coordinates	
print(f"Min x = {min_px}, Max x = {max_px}, \nMin y = {min_py}, Max y = {max_py}")

# %% [markdown]
# ### Select a monthly stack of Sentinel 2 bands
# 
# We will select a monthly stack of Sentinel 2 bands to predict the habitat selection over. First use a function that will select the correct stack of bands for a given month.
# 
# This indexing is slightly different from the indexing we used for the `deepSSF_landscape_preds.ipynb` and `deepSSF_simulations.ipynb` notebooks, which was indexing NDVI layers. In that case we were indexing the layers directly, and therefore the first entry was at 0 (i.e., March was in month_index = 2). Here, we are creating a string that corresponds to the layer name, and therefore the first entry is at 1. (i.e., March will be at month_index = 3)

# %%
# Create a mapping from day of the year to month index
def day_to_month_index(day_of_year):
    # Calculate the year and the day within that year
    base_date = datetime(2019, 1, 1)
    date = base_date + timedelta(days=int(day_of_year) - 1)
    year_diff = date.year - base_date.year
    month_index = (date.month) + (year_diff * 12)  # month index (1-based)
    if month_index == 0:
        month_index += 1
    return month_index

# %% [markdown]
# Choose a day of the year and get the month to index by.

# %%
# Choose a day of the year
yday_t1 = 50
# Get the month index for the selected day
month_index = day_to_month_index(yday_t1) 
print(f'Month index:    {month_index}')

# For sentinel 2 data
selected_month = f'2019_{month_index:02d}'
print(f'Selected month: {selected_month}')

# Get the data for the selected month
sentinel_layers = data_dict[selected_month]
print(sentinel_layers.shape)

# %% [markdown]
# # Subset all layers
# 
# ### Create directory for saving plots

# %%
# Output directory for saving plots
landscape_preds_dir = f'{output_dir}/landscape_preds'
os.makedirs(landscape_preds_dir, exist_ok=True)
print(f"Output directory: {landscape_preds_dir}")

# %% [markdown]
# Here we want to crop the layers to the spatial extent defined above. 
# 
# For plotting purposes, we convert the normalised landscape layers back to their natural scale. This is only for plotting and are not used for modelling. 
# 
# We also calculate the minimum and maximum values of the normalised landscape subsets, which we will use to define the colour scale when plotting the layers.

# %%
# Initialize a subset array with zeros (or another padding value) 
# with the dimensions defined by the pixel indices.
subset = np.zeros((min_py - max_py, max_px - min_px), dtype=slope_landscape.dtype)

# Initialize a list to store the results
layer_subsets = []

# Loop over each layer in sentinel_layers and global_raster_tensors
for sentinel_layer in sentinel_layers:
    # Process the sentinel layer
    sentinel_layer = torch.from_numpy(sentinel_layer)
    # print(sentinel_layer.shape)
    sentinel_result = sentinel_layer[max_py:min_py, min_px:max_px]
    layer_subsets.append(sentinel_result)

# Add the normalised slope layer to the list of layer subsets
slope_subset = slope_landscape_norm[max_py:min_py, min_px:max_px]
# Convert the slope subset back to the natural scale (for plotting only)
slope_subset_natural = slope_subset * (slope_max - slope_min) + slope_min

# Append the slope subset to the list of layer subsets
layer_subsets.append(slope_subset)

# Pull out the Sentinel-2 RGB bands for plotting
r_band = layer_subsets[1].detach().numpy()
g_band = layer_subsets[2].detach().numpy()
b_band = layer_subsets[3].detach().numpy()

# Stack the bands along a new axis
rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

# # Apply gamma correction to brighten the image
gamma = 1.1
rgb_image = rgb_image ** (1/gamma)

# Plot and save the RGB image
plt.imshow(rgb_image)
plt.title('Sentinel 2 RGB')
plt.savefig(f"{landscape_preds_dir}/id{buffalo_id}_rgb_yday{yday}.png", dpi=600, bbox_inches='tight')
plt.show(block=False)
plt.close()  # Close the figure to free memory

# plot the subset
plt.imshow(slope_subset_natural, cmap='viridis')
# plt.colorbar(shrink=0.7)
plt.title(f'Slope')
plt.savefig(f"{landscape_preds_dir}/id{buffalo_id}_slope.png", dpi=600, bbox_inches='tight')
plt.show(block=False)
plt.close()  # Close the figure to free memory


# %% [markdown]
# ## Prepare the scalar covariates
# 
# We need the hour of the day and day of the year to predict with (as the model was trained with these as inputs). 
# 
# In this example we'll select a single hour and a single day of the year to predict over, and then below we'll loop over the hours of the day to illustrate the temporal variation in habitat selection that was learned by the model.

# %%
# Hour of the day (hour) 
hour_t1 = 17
# Convert to sine and cosine (as this is what the model expects)
hour_t1_sin = np.sin(2*np.pi*hour_t1/24)
hour_t1_cos = np.cos(2*np.pi*hour_t1/24)

# Day of the year (yday)
print(f"Day of the year: {yday}") # yday was defined earlier
# Convert to sine and cosine
yday_t1_sin = np.sin(2*np.pi*yday_t1/365)
yday_t1_cos = np.cos(2*np.pi*yday_t1/365)

# Convert the numpy scalars to PyTorch tensors and ensure they are float type.
# unsqueeze(0) adds a batch dimension.
hour_t1_sin_tensor = torch.tensor(hour_t1_sin).float().unsqueeze(0)
hour_t1_cos_tensor = torch.tensor(hour_t1_cos).float().unsqueeze(0)
yday_t1_sin_tensor = torch.tensor(yday_t1_sin).float().unsqueeze(0)
yday_t1_cos_tensor = torch.tensor(yday_t1_cos).float().unsqueeze(0)

def scalar_to_grid(x, dim_x, dim_y):
    """
    Reshape a scalar tensor to a grid with the given dimensions.
    
    Args:
        x (torch.Tensor): Input scalar tensor of shape [1].
        dim_x (int): Number of rows for the grid.
        dim_y (int): Number of columns for the grid.
        
    Returns:
        torch.Tensor: Tensor expanded to shape [1, 1, dim_x, dim_y].
    """
    # View x as shape [1, 1, 1, 1] and expand to a grid of shape [1, 1, dim_x, dim_y]
    scalar_map = x.view(1, 1, 1, 1).expand(1, 1, dim_x, dim_y)
    return scalar_map

# Stack the scalar tensors along a new dimension (dim=0) to form a tensor with shape [4, 1]
# representing the four covariates: hour sin, hour cos, day-of-year sin, day-of-year cos.
scalar_covariates = torch.stack([
    hour_t1_sin_tensor, 
    hour_t1_cos_tensor, 
    yday_t1_sin_tensor, 
    yday_t1_cos_tensor
], dim=0)
print(scalar_covariates.shape)  # Expected shape: [4, 1]

# Convert each scalar covariate into a grid matching the dimensions of the NDVI landscape.
# This creates spatial maps where each grid cell contains the same scalar value.
scalar_grids = torch.cat([
    scalar_to_grid(tensor, slope_subset.shape[0], slope_subset.shape[1]) 
    for tensor in scalar_covariates
], dim=1)
print(scalar_grids.shape)  # Expected shape: [1, 4, rows, cols]

# %% [markdown]
# ## Combine the spatial and scalar (as grid) covariates

# %%
# Stack the individual landscape layers into a single tensor.
# The resulting tensor has shape [13, rows, cols].
landscape_stack = torch.stack(layer_subsets, dim=0)

# Add an extra batch dimension to the tensor.
# Now the shape becomes [1, 4, rows, cols], where 1 indicates a single sample.
landscape_stack = landscape_stack.unsqueeze(0)
print(landscape_stack.shape)

# Concatenate the landscape stack with the scalar grids along the channel dimension.
# This merges the spatial features (landscape_stack) and the repeated scalar covariate grids (scalar_grids)
# into a full feature tensor with shape [1, total_channels, rows, cols].
full_stack = torch.cat([landscape_stack, scalar_grids], dim=1).to(device)
print(full_stack.shape)

# %% [markdown]
# ## Run habitat selection subnetwork on the landscape layers
# 
# All we need to do to run the habitat selection subnetwork of the model is to pull that component out of the model and run it on the landscape layers.

# %%
# Pass the full feature stack through the habitat convolutional layer of the model to get predictions.
landscape_predictions = model.conv_habitat(full_stack)

# Print the shape of the output tensor to verify the dimensions of the predictions.
print(landscape_predictions.shape)

# %% [markdown]
# ## Plot the predictions
# 
# Create a directory with a folder for the day of the year.

# %%
# To save the images, create a directory
landscape_preds_day_dir = f'{landscape_preds_dir}/yday{yday_t1}'
os.makedirs(landscape_preds_day_dir, exist_ok=True)
print(f"Output directory: {landscape_preds_day_dir}")

# %% [markdown]
# As the spatial inputs are padded by the model, there are artifacts at the edges of the predictions. Sometimes this can result in quite different values to the rest of the predictions, which changes the colour scale. To prevent this we remove the outer pixels of the predictions.

# %%
# Extract the first sample from the output tensor, detach it from the computational graph,
# move it to the CPU, and convert it to a NumPy array for further processing and visualization.
output_image = landscape_predictions[0].detach().cpu().numpy()

# Create masks for the x and y coordinates, as well as a water mask (unused in the code below),
# with the same shape as the output image.
x_mask = np.ones_like(output_image)
y_mask = np.ones_like(output_image)
water_mask = np.ones_like(output_image)

# Get the dimensions of the output image.
y_dim = output_image.shape[0]
x_dim = output_image.shape[1]

# Define a buffer value to mask out edge cells.
buffer = 3

# Apply the buffer mask to the x-axis: set the first and last 'buffer' columns to -infinity.
x_mask[:, :buffer] = -np.inf
x_mask[:, x_dim - buffer:] = -np.inf

# Apply the buffer mask to the y-axis: set the first and last 'buffer' rows to -infinity.
y_mask[:buffer, :] = -np.inf
y_mask[y_dim - buffer:, :] = -np.inf

# Mask out edge cells in the output image by multiplying with the x and y masks.
# Also mask out water cells by multiplying with the water mask.
output_image = output_image * x_mask * y_mask
output_image = output_image * water_mask

# Plot the masked output image using the 'viridis' colormap.
plt.imshow(output_image, cmap='viridis')
plt.colorbar(shrink=0.7)  # Display the color scale.
plt.title(f'Log-probabilities: Day {yday_t1}, Hour {hour_t1}')
# Define the filename for saving the landscape prediction image
# plt.savefig(f"{landscape_preds_day_dir}/id{buffalo_id}_hab_log_prob_yday{yday_t1}_hour{hour_t1}.png", 
#             dpi=600, bbox_inches='tight') 
plt.show(block=False)
plt.close()  # Close the figure to free up memory.

# Plot the exponential of the output image, which may be used to convert log-probabilities
# back to probability values.
plt.imshow(np.exp(output_image), cmap='viridis')
plt.colorbar(shrink=0.7)  # Display the color scale.
plt.title(f'Probabilities: Day {yday_t1}, Hour {hour_t1}')
# Define the filename for saving the landscape prediction image
# plt.savefig(f"{landscape_preds_day_dir}/id{buffalo_id}_hab_prob_yday{yday_t1}_hour{hour_t1}.png", 
            # dpi=600, bbox_inches='tight') 
plt.show(block=False)
plt.close()  # Close the figure to free up memory.

# %% [markdown]
# ## Loop over hours
# 
# A benefit of the `deepSSF` approach is that it can represent temporal variation in habitat selection (and movement dynamics) across the day, which interacts with the day of the year. 
# 
# To illustrate this, we can loop over the hours of the day and predict the habitat selection at each hour. We can also assess the contribution of each covariate to the predictions at each hour, giving us some idea of what the model has learned about the temporal variation in habitat selection, which can help us to further understand our species' spatial ecology.

# %%
# To plot the prediction maps with the same colour scale, 
# we need to determine the minimum and maximum values
# Initialize landscape min and max values
landscape_vmin = float('inf')
landscape_vmax = float('-inf')

# %% [markdown]
# ### Select the day of the year
# 
# We will select a single day of the year to predict over, and then loop over the hours of the day.

# %%
# Choose a day of the year
yday_t1 = 130
# Get the month index for the selected day
month_index = day_to_month_index(yday_t1) 
print(f'Month index:    {month_index}')

# For sentinel 2 data
selected_month = f'2019_{month_index:02d}'
print(f'Selected month: {selected_month}')

# Get the data for the selected month
sentinel_layers = data_dict[selected_month]
print(sentinel_layers.shape)

# Convert day of the year to sine and cosine
yday_t1_sin = np.sin(2*np.pi*yday_t1/365)
yday_t1_cos = np.cos(2*np.pi*yday_t1/365)

# %% [markdown]
# ## Crop the Sentinel-2 bands

# %%
# Initialize a subset array with zeros (or another padding value) 
# with the dimensions defined by the pixel indices.
subset = np.zeros((min_py - max_py, max_px - min_px), dtype=slope_landscape.dtype)

# Initialize a list to store the results
layer_subsets = []

# Loop over each layer in sentinel_layers and global_raster_tensors
for sentinel_layer in sentinel_layers:
    # Process the sentinel layer
    sentinel_layer = torch.from_numpy(sentinel_layer)
    # print(sentinel_layer.shape)
    sentinel_result = sentinel_layer[max_py:min_py, min_px:max_px]
    layer_subsets.append(sentinel_result)

# Append the slope subset to the list of layer subsets
layer_subsets.append(slope_subset)

# Stack the spatial layers along a new dimension (dim=0)
landscape_stack = torch.stack(layer_subsets, dim=0)
landscape_stack = landscape_stack.unsqueeze(0) # add a batch dimension
print(landscape_stack.shape) # Expected shape: [1, 13, rows, cols]

# %% [markdown]
# Update the directory with the correct day of the year.

# %%
landscape_preds_day_dir_log = f'{landscape_preds_day_dir}/log_probabilities'
os.makedirs(landscape_preds_day_dir_log, exist_ok=True)
print(f"Output directory: {landscape_preds_day_dir_log}")

landscape_preds_day_dir_probs = f'{landscape_preds_day_dir}/probabilities'
os.makedirs(landscape_preds_day_dir_probs, exist_ok=True)
print(f"Output directory: {landscape_preds_day_dir_probs}")

# %% [markdown]
# As we need to run over the full loop to get the `landscape_vmin` and `landscape_vmax` values, we will first run the loop over the hours of the day once without plotting (which is much faster), and then in the following code chunk run over the loop again but plot the predictions and save the images.
# 
# This loop generates the predictions and saves the probability values as a csv to correlate with covariate values (which we do in R).

# %%
# Define the range of hours you want to loop over
hours = range(1,25) # to start at 1

# As we used sine and cosine terms to represent the hour of the day,
# rather than the hour as integers, we can use a continuous range of hours
# (Uncomment line below)
# hours = np.arange(0,24, 0.1)

for hour_t1 in hours:

    # convert hour to sine and cosine
    hour_t1_sin = np.sin(2*np.pi*hour_t1/24)
    hour_t1_cos = np.cos(2*np.pi*hour_t1/24)

    # Convert numpy objects to PyTorch tensors
    hour_t1_sin_tensor = torch.tensor(hour_t1_sin).float().unsqueeze(0)
    hour_t1_cos_tensor = torch.tensor(hour_t1_cos).float().unsqueeze(0)
    yday_t1_sin_tensor = torch.tensor(yday_t1_sin).float().unsqueeze(0)
    yday_t1_cos_tensor = torch.tensor(yday_t1_cos).float().unsqueeze(0)

    # Stack tensors column-wise to create a tensor of shape 
    scalar_covariates = torch.stack([hour_t1_sin_tensor, 
                                     hour_t1_cos_tensor, 
                                     yday_t1_sin_tensor, 
                                     yday_t1_cos_tensor], 
                                     dim=0)
    
    # Convert each scalar covariate into a grid matching the dimensions of the NDVI landscape
    scalar_grids = torch.cat([scalar_to_grid(tensor, 
                                             slope_subset.shape[0], 
                                             slope_subset.shape[1]) 
                                             for tensor in scalar_covariates
                                             ], dim=1)

    # Stack the spatial (landscape_stack) and scalar covariates (as grids)
    full_stack = torch.cat([landscape_stack, scalar_grids], dim=1).to(device)
    # print(full_stack.shape) # Expected shape: [1, n_channels, rows, cols]

    # Run the model
    landscape_predictions = model.conv_habitat(full_stack)
    # print(landscape_predictions.shape)

    # Pull out the prediction
    output_image = landscape_predictions[0].detach().cpu().numpy()

    # Mask out cells on the edges (that affect the colour scale)
    output_image = output_image * x_mask * y_mask

    # Check if output_image is valid before updating landscape min and max
    if output_image.size > 0:
        # Ignore masked values in the calculation
        valid_values = output_image[np.isfinite(output_image)]
        if valid_values.size > 0:
            current_min = valid_values.min()
            current_max = valid_values.max()

            # Update landscape min and max values for scaling the colour map
            landscape_vmin = min(landscape_vmin, current_min)
            landscape_vmax = max(landscape_vmax, current_max)

print(landscape_vmin, landscape_vmax)

# %%
for hour_t1 in hours:

    # convert hour to sine and cosine
    hour_t1_sin = np.sin(2*np.pi*hour_t1/24)
    hour_t1_cos = np.cos(2*np.pi*hour_t1/24)

    # Convert numpy objects to PyTorch tensors
    hour_t1_sin_tensor = torch.tensor(hour_t1_sin).float().unsqueeze(0)
    hour_t1_cos_tensor = torch.tensor(hour_t1_cos).float().unsqueeze(0)
    yday_t1_sin_tensor = torch.tensor(yday_t1_sin).float().unsqueeze(0)
    yday_t1_cos_tensor = torch.tensor(yday_t1_cos).float().unsqueeze(0)

    # Stack tensors column-wise to create a tensor of shape 
    scalar_covariates = torch.stack([hour_t1_sin_tensor, 
                                     hour_t1_cos_tensor, 
                                     yday_t1_sin_tensor, 
                                     yday_t1_cos_tensor], 
                                     dim=0)
    
    # Convert each scalar covariate into a grid matching the dimensions of the NDVI landscape
    scalar_grids = torch.cat([scalar_to_grid(tensor, 
                                             slope_subset.shape[0], 
                                             slope_subset.shape[1]) 
                                             for tensor in scalar_covariates
                                             ], dim=1)

    # Stack the spatial (landscape_stack) and scalar covariates (as grids)
    full_stack = torch.cat([landscape_stack, scalar_grids], dim=1).to(device)
    # print(full_stack.shape) # Expected shape: [1, n_channels, rows, cols]

    # Run the model
    landscape_predictions = model.conv_habitat(full_stack)
    # print(landscape_predictions.shape)

    # Pull out the prediction
    output_image = landscape_predictions[0].detach().cpu().numpy()

    # Mask out cells on the edges (that affect the colour scale)
    output_image = output_image * x_mask * y_mask

    # Habitat selection log-probabilities
    filename_landscape_preds = f"{landscape_preds_day_dir_log}/id{buffalo_id}_log_hab_sel_yday{yday_t1}_hour{hour_t1}.png"
    plt.figure()  # Create a new figure
    plt.imshow(output_image)#, vmin=landscape_vmin, vmax=landscape_vmax)
    plt.colorbar(shrink=0.7)
    plt.title(f'Log-Probabilities: Day {yday_t1}, Hour {hour_t1}')
    plt.savefig(filename_landscape_preds, dpi=300)#, bbox_inches='tight')
    # plt.show(block=False)
    plt.close()  # Close the figure to free memory

    # Habitat selection probabilities
    filename_landscape_preds = f"{landscape_preds_day_dir_probs}/id{buffalo_id}_hab_sel_yday{yday_t1}_hour{hour_t1}.png"
    plt.figure()  # Create a new figure
    plt.imshow(np.exp(output_image))#, vmin=np.exp(landscape_vmin), vmax=np.exp(landscape_vmax))
    plt.colorbar(shrink=0.7)
    plt.title(f'Probabilities: Day {yday_t1}, Hour {hour_t1}')
    plt.savefig(filename_landscape_preds, dpi=300)#, bbox_inches='tight')
    # plt.show(block=False)
    plt.close()  # Close the figure to free memory


# %% [markdown]
# ### Make a GIF of the landscape predictions
# 
# First, here's a function to call to make a gif from a given directory.

# %%
# Example sorting by the epoch number
def extract_hour(filename):
    # Extract the epoch number from the filename
    # Adjust the extraction based on your naming pattern
    import re
    match = re.search(r'hour(\d+).', filename)
    if match:
        return int(match.group(1))
    return 0

def create_gif(image_folder, output_filename, fps=10):
    """
    Creates a GIF from a sequence of images in a folder.

    Parameters:
    - image_folder: Path to the folder containing images
    - output_filename: Name of the output GIF file
    - duration: Duration of each frame in seconds
    """
    # Get all png files in the specified folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=extract_hour)

    # Check if any images were found
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Read all images
    frames = [imageio.imread(image) for image in images]

    # Save as GIF
    imageio.mimsave(output_filename, frames, fps=fps, loop=0)

    display(Image(filename=output_filename))

    print(f"GIF created successfully: {output_filename}")


# %% [markdown]
# ## GIF of log-probabilities

# %%
# Path to your images
image_folder =  f'{landscape_preds_day_dir_log}'
# Output GIF filename
output_filename = f'{landscape_preds_day_dir_log}/../../landscape_predictions_log_probabilities_{yday_t1}.gif'
# Create the GIF
create_gif(image_folder, output_filename, fps=5)

# %% [markdown]
# ## GIF of probabilities

# %%
# Path to your images
image_folder =  f'{landscape_preds_day_dir_probs}'
# Output GIF filename
output_filename = f'{landscape_preds_day_dir_probs}/../../landscape_predictions_probabilities_{yday_t1}.gif'
# Create the GIF
create_gif(image_folder, output_filename, fps=5)

# %% [markdown]
# # Generate simulations

# %% [markdown]
# # Next-step function
# 
# This function is slightly different from the one we used in the `deepSSF_simulations.ipynb` notebook, mainly due to the indexing and cropping of the Sentinel-2 layers.
# 
# The next-step function will take the following inputs:
# - environmental rasters at the landscape level,
# - the scalar covariates to be turned into grids (temporal covariates in our case),
# - the previous bearing to predict the turning angle from,
# - the size of the local layer to crop out,
# - the current location of the simulated individual,
# - the raster transformation to convert between pixel and geographic coordinates.
# 
# **Adding jitter to the next step**  
# As location of the next step is a particular cell (which are 25 m x 25 m), we also have an additional component in the function that adds an element of randomness to where in the cell the next step is. This prevents the location of the next step from being in exactly the same location as the current step (which would lead to a null bearing and 0 step length), and prevents an artificial grid-like pattern from emerging in the simulated trajectories. As all of the probability values are exactly the same within a cell, this is analogous to the process of simulating from an SSF, where continuous step lengths and turning angles are drawn from their distributions, which land somewhere in a particular cell. However, the difference is that the probability values for the movement kernel in an SSF are still continuous, whereas in the deepSSF model they are discrete (due to calculating the movement probability for each cell). This may lead to artifacts for steps very close to the current location, as the Gamma distribution has high probability near 0.
# 
# We 'jitter' the point by adding a small amount of noise in the x and y direction. This small amount is drawn from a 2D (uncorrelated) normal distribution with a mean of 0 and a standard deviation of 6.5 m, such that about 95% of the jittered points will fall within the 25 m x 25 m cell. If the jittered point falls outside of the cell, we re-sample the jittered point until it falls within the cell. 
# 
# This could have also been a uniform distribution between -12.5 and 12.5 in the x and y directions, but we chose a normal distribution as for the centre cell we still want it to be close to 0. This was a modelling choice and different choices here will have slightly different effects (although mainly on the step length and turning angle distributions). The cell that was selected, which becomes the cell to start the next step from, will be the same.

# %%
def simulate_next_step(sentinel_data_dict,
                        which_month,
                        landscape_raster_tensors, 
                        scalars_to_grid,
                        bearing,
                        window_size, 
                        x_loc, 
                        y_loc,
                        landscape_raster_transform):

    # for sentinel 2 data
    selected_month = f'2019_{which_month:02d}'
    # Get the normalized data for the selected month
    sentinel_layers = sentinel_data_dict[selected_month]
    # print(sentinel_layers.shape)

    # Initialize a list to store the results
    results = []

    # Loop over each layer in sentinel_layers and landscape_raster_tensors
    for sentinel_layer in sentinel_layers:
        # Process the sentinel layer
        sentinel_layer = torch.from_numpy(sentinel_layer)
        # print(sentinel_layer.shape)
        sentinel_result = subset_function(sentinel_layer, 
                                          x=x_loc, 
                                          y=y_loc, 
                                          window_size=window_size, 
                                          transform=landscape_raster_transform)
        results.append(sentinel_result)
    
    for raster_tensor in landscape_raster_tensors:
        # Process the landscape raster tensor
        raster_result = subset_function(raster_tensor, 
                                        x=x_loc, 
                                        y=y_loc, 
                                        window_size=window_size, 
                                        transform=landscape_raster_transform)
        # Append the slope to the Sentinel-2 layers
        results.append(raster_result)
    
    # Unpacking the results
    subset_rasters_tensors, origin_xs, origin_ys = zip(*results)

    # Pull out the RGB bands
    s2_b2_subset = subset_rasters_tensors[1]
    s2_b3_subset = subset_rasters_tensors[2]
    s2_b4_subset = subset_rasters_tensors[3]

    # Pull out the slope
    slope_subset = subset_rasters_tensors[12]
    
    # Stack the processed tensors along a new dimension (e.g., dimension 0)
    x1 = torch.stack(subset_rasters_tensors, dim=0)
    x1 = x1.unsqueeze(0).to(device)
    # print(x1.shape)

    # create masking layer to remove outside of the extent
    # where the value is -1, set to NaN (to be masked)
    single_layer = x1[0, 0, :, :]
    mask = torch.where(single_layer == -1, torch.tensor(float('nan')), 1)

    # Scalar data to be converted to a grid
    x2 = scalars_to_grid.to(device)

    # Bearing data (initialised to 0 but updated at each simulated step)
    x3 = bearing.to(device)

    # Run the model
    model_output = model((x1, x2, x3))

    # Extract the habitat and movement log probabilities
    hab_log_prob = model_output[:, :, :, 0]
    move_log_prob = model_output[:, :, :, 1]

    # Combine the habitat and movement log probabilities
    step_log_prob = (hab_log_prob + move_log_prob)
    step_log_prob = step_log_prob * mask

    hab_log_prob = hab_log_prob.squeeze()
    move_log_prob = move_log_prob.squeeze()
    step_log_prob = step_log_prob.squeeze()

    # sample from the array values
    step_prob = torch.exp(step_log_prob)
    step_prob = torch.nan_to_num(step_prob, nan=0.)
    step_prob_norm = step_prob/torch.sum(step_prob)

    # Flatten the probability surface
    flat_step_prob_norm = step_prob_norm.flatten().detach().cpu().numpy()
    # print(flat_prob_surface)

    # Generate the corresponding indices for the flattened array
    indices = np.arange(flat_step_prob_norm.size)

    # Sample from the flattened probability surface
    sampled_index = np.random.choice(indices, p=flat_step_prob_norm)

    # Convert the sampled index back to 2D coordinates
    sampled_coordinates = np.unravel_index(sampled_index, step_prob_norm.shape)

    # THE Y COORDINATE COMES FIRST in the sampled coordinates
    new_px = origin_xs[0] + sampled_coordinates[1]
    new_py = origin_ys[0] + sampled_coordinates[0]

    # Convert geographic coordinates to pixel coordinates
    new_x, new_y = raster_transform * (new_px, new_py)

    # Sample from a normal distribution with mean 0 and sd 6.5, 
    # which are the parameters where the cell contains ~ 95% of density
    # if it's outside the bounds of the cell, resample
    while True:
        jitter_x = np.random.normal(0, 6.5)
        if -12.5 <= jitter_x <= 12.5:
            break

    # Sample jitter for new_y and ensure it is within bounds
    while True:
        jitter_y = np.random.normal(0, 6.5)
        if -12.5 <= jitter_y <= 12.5:
            break

    # Add the valid jitter to new_x and new_y
    new_x = new_x + jitter_x
    new_y = new_y + jitter_y

    # print(new_x, new_y)

    # Return the new_x and new_y coordinates, 
    # the probability surfaces for optional plotting,
    # and the sampled coordinates (next step in local layer pixel coordinates)
    return new_x, \
           new_y, \
           hab_log_prob, \
           move_log_prob, \
           step_log_prob, \
           sampled_coordinates[1], \
           sampled_coordinates[0], \
           s2_b2_subset, \
           s2_b3_subset, \
           s2_b4_subset, \
           slope_subset


# %% [markdown]
# ## Test the function

# %%
# Rasters besides the Sentinel-2 bands
landscape_raster_list = [slope_landscape_norm]

x2 = torch.stack((torch.tensor(buffalo_df['hour_t1_sin1'].iloc[step_index]).float(), 
                  torch.tensor(buffalo_df['hour_t1_cos1'].iloc[step_index]).float(),
                  torch.tensor(buffalo_df['yday_t1_sin1'].iloc[step_index]).float(),
                  torch.tensor(buffalo_df['yday_t1_cos1'].iloc[step_index]).float()), dim=0).unsqueeze(0)

# Debugging prints
# print(x2)
# print(x2.shape)

# recover the hour value for the step
hour_t1 = int(deepSSF_utils.recover_hour(buffalo_df['hour_t1_sin1'].iloc[step_index], 
                                         buffalo_df['hour_t1_cos1'].iloc[step_index]))
print(f'Hour of the day:                                    {hour_t1}')

# recover the day of the year value for the step
yday_t1 = int(deepSSF_utils.recover_yday(buffalo_df['yday_t1_sin1'].iloc[step_index], 
                                         buffalo_df['yday_t1_cos1'].iloc[step_index]))
print(f'Day of the year:                                    {yday_t1}')

# Pull out the bearing
bearing = buffalo_df['bearing'].iloc[step_index]
bearing_step = torch.tensor(bearing).float().unsqueeze(0).unsqueeze(0)
print(f'Bearing at the step:                                {round(bearing, 3)}')

# Debugging prints
# print(bearing_step)
# print(bearing_step.shape)

# Simulate the next step
test_outputs = simulate_next_step(sentinel_data_dict=data_dict,
                                    which_month=month_index,
                                    landscape_raster_tensors=landscape_raster_list,
                                    scalars_to_grid=x2,
                                    bearing=bearing_step,
                                    window_size=window_size,
                                    x_loc=x, # x location defined above
                                    y_loc=y, # y location defined above
                                    landscape_raster_transform=raster_transform)

# Unpack the test outputs
(new_x, new_y, 
 hab_log_prob, move_log_prob, step_log_prob, 
 px, py, 
 s2_b2_subset, s2_b3_subset, s2_b4_subset, 
 slope_subset) = test_outputs

print(f'New location in local layer pixel coordinates:      {px, py}')
print(f'New location in geographic coordinates:             {new_x, new_y}')

# Create the mask for edge cells 
# (as the convolution filters create artifacts due to the padding)
x_mask = np.ones_like(hab_log_prob.detach().cpu().numpy())
y_mask = np.ones_like(hab_log_prob.detach().cpu().numpy())

# mask out cells on the edges that affect the colour scale
x_mask[:, :3] = -np.inf
x_mask[:, window_size-3:] = -np.inf
y_mask[:3, :] = -np.inf
y_mask[window_size-3:, :] = -np.inf

# Set the pixel of the next step, which is at (px, py) to -inf
hab_log_prob[(px, py)] = -np.inf
move_log_prob[(px, py)] = -np.inf
step_log_prob[(px, py)] = -np.inf

# Plot the RGB bands
# Grab the RGB bands from the model output above
r_band = s2_b4_subset.detach().numpy()
g_band = s2_b3_subset.detach().numpy()
b_band = s2_b2_subset.detach().numpy()

# Stack the bands along a new axis
rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

plt.figure()  # Create a new figure
plt.imshow(rgb_image)
plt.title('Sentinel 2 RGB')
plt.show(block=False)
plt.close()  

plt.figure()  # Create a new figure
plt.imshow(slope_subset.detach().numpy())
plt.title('Slope')
plt.show(block=False)
plt.close()  

# Plot the habitat selection log-probability after applying the border mask
plt.imshow(hab_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask)
plt.colorbar()
plt.title('Habitat selection log-probability')
plt.show(block=False)
plt.close()  

# Plot the movement log-probability
plt.imshow(move_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask)
plt.colorbar()
plt.title('Movement log-probability')
plt.show(block=False)
plt.close()  

# Plot the next step log-probability with the masks applied.
plt.imshow(step_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask)
plt.colorbar()
plt.title('Next step log-probability')
plt.show(block=False)
plt.close()  

# %% [markdown]
# # Generate trajectory
# 
# Now we can loop over the next-step function to generate a trajectory.

# %% [markdown]
# ## Setup parameters

# %%
# Setup the simulation parameters
n_steps = 3000

# Starting location and yday of buffalo 2005 (training data)
start_x = buffalo_df['x1_'].iloc[0].astype(np.float32)
start_y = buffalo_df['y1_'].iloc[0].astype(np.float32)
starting_yday = buffalo_df['yday_t1'].iloc[0].astype(np.float32)

print(f'Starting x and y coordinates: {start_x}, {start_y}')
print(f'Starting day of the year:     {starting_yday}')

landscape_raster_list = [slope_landscape_norm]
landscape_raster_transform = raster_transform

# output directory for saving plots
output_dir = f'outputs/deepSSF_prob_maps/S2/{buffalo_id}'
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Create simulation inputs from the parameters
# 
# Create empty lists to store the simulated locations and bearings, and initialise the vectors of temporal covariates.

# %%
# Empty lists to store the x and y coordinates
x = np.repeat(0., n_steps).astype(np.float32)
y = np.repeat(0., n_steps).astype(np.float32)

# Set the first entry as the starting location
x[0], y[0] = start_x, start_y

# Create an hour-of-day sequence and repeat it until it reaches n_steps.
hour_t1 = np.resize(range(24), n_steps)

# Convert hour-of-day values into sine and cosine components.
hour_t1_sin = np.sin(2 * np.pi * hour_t1 / 24).astype(np.float32)
hour_t1_cos = np.cos(2 * np.pi * hour_t1 / 24).astype(np.float32)

# Create the day of the year sequences 
yday_t1 = np.repeat(range(int(starting_yday), int(starting_yday) + 365), 24)
yday_t1 = np.resize(yday_t1, n_steps)

# Convert day-of-year values into sine and cosine components.
yday_t1_sin = np.sin(2 * np.pi * yday_t1 / 365).astype(np.float32)
yday_t1_cos = np.cos(2 * np.pi * yday_t1 / 365).astype(np.float32)

# Initialise a bearing vector with zeroes for all simulation steps
bearing = np.repeat(0., n_steps).astype(np.float32)

# Convert lists to PyTorch tensors
hour_t1_tensor = torch.tensor(hour_t1).float()
hour_t1_sin_tensor = torch.tensor(hour_t1_sin).float()
hour_t1_cos_tensor = torch.tensor(hour_t1_cos).float()
yday_t1_tensor = torch.tensor(yday_t1).float()
yday_t1_sin_tensor = torch.tensor(yday_t1_sin).float()
yday_t1_cos_tensor = torch.tensor(yday_t1_cos).float()  
bearing_tensor = torch.tensor(bearing).float()

# Stack tensors column-wise to create a tensor of shape [n_steps, 4]
x2_full = torch.stack((hour_t1_sin_tensor, 
                       hour_t1_cos_tensor, 
                       yday_t1_sin_tensor, 
                       yday_t1_cos_tensor), dim=1)

# Initialize variables to cache the previous yday and month index
previous_yday = None

# %% [markdown]
# # Trajectory loop
# 
# Now we can loop over the next-step function to generate a trajectory. Essentially we have to a some starting location, which goes through the `simulate_next_step` function to get the next location, which then becomes the current location for the next step.
# 
# For indexing, the location we are trying to predict (the next step) is index `i`, and the current location is index `i-1`. This prevents the loop from breaking by storing the final location at `i+1`, which is outside of the loop range.
# 
# The bearing at `i-1` is the bearing between `i-1` and `i-2`, i.e., the bearing that location `i-1` was approached from.
# 
# After the `simulate_next_step` function, we have some code for generating and saving plots of the habitat selection, movement and next-step predictions (all on the log-scale as they're more informative). This is helpful to check that everything is working as expected, but can also be used for making animations (when plotting the surfaces for the whole trajectory). When making a trajectory it can be helpful to comment out the `plt.show(block=False)` lines, so the plots will just save to file and not pop up.

# %%
for i in range(1, n_steps):

    x_loc = x[i-1]
    y_loc = y[i-1]

    # calculate the bearing from the previous location
    if i > 1:
        bearing_rad = np.arctan2(y[i-1] - y[i-2], x[i-1] - x[i-2])
    else:
        # if it's the first step, sample a random bearing
        bearing_rad = np.random.uniform(-np.pi, np.pi)

    # Store the bearing in the vector
    bearing[i-1] = bearing_rad
    # print("Bearing[i-1]", bearing[i-1])

    # Convert the bearing to a tensor and add dimensions for the batch and channel
    bearing_tensor = torch.tensor(bearing[i-1]).unsqueeze(0).unsqueeze(0)
    # print(bearing_tensor.shape) # Debugging print

    # Select the temporal features for the specific step
    x2 = x2_full[i-1,:].unsqueeze(dim=0)
    # print(x2)
    
    # Determine the month index based on the day of the year
    day_of_year = yday_t1[i-1] % 365
    if day_of_year != previous_yday:
        month_index = day_to_month_index(day_of_year)
        previous_yday = day_of_year

    # print(f'Day of the year: {day_of_year}') # Debugging print
    # print(f'Month index: {month_index}') # Debugging print

    # Landscape rasters besides the Sentinel-2 bands
    landscape_raster_list = [slope_landscape_norm]
    
    # Run the simulation for the next step
    sim_outputs = simulate_next_step(sentinel_data_dict=data_dict,
                                      which_month=month_index,
                                      landscape_raster_tensors=landscape_raster_list,
                                      scalars_to_grid=x2,
                                      bearing=bearing_tensor,
                                      window_size=window_size,
                                      x_loc=x_loc,
                                      y_loc=y_loc,
                                      landscape_raster_transform=landscape_raster_transform)
    
    (new_x, new_y, 
     hab_log_prob, move_log_prob, step_log_prob, 
     px, py, 
     s2_b2, s2_b3, s2_b4,
     slope_subset) = sim_outputs
    # print(f'New location in pixel coordinates           {px, py}') # Debugging print
    # print(f'New location in geographic coordinates      {new_x, new_y}\n') # Debugging print

    x[i] = new_x
    y[i] = new_y
    
    
    # -------------------------------------------------------------------------
    # Plot the probability surfaces for habitat selection, movement, and next step
    # -------------------------------------------------------------------------

    # The x_mask and y_mask objects should have already been defined earlier in the code

    # Plot the first few probability surfaces - change the condition to i < n_steps to plot all
    if i < 250:
        
        # pull out the RGB bands
        r_band = s2_b4.detach().numpy()
        g_band = s2_b3.detach().numpy()
        b_band = s2_b2.detach().numpy()

        # Stack the bands along a new axis
        rgb_image = np.stack([r_band, g_band, b_band], axis=-1)
        # Normalize to the range [0, 1] for display
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())


        ## Habitat probability surface
        hab_log_prob = hab_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask
        hab_log_prob[py, px] = -np.inf

        ## Movement probability surface
        move_log_prob = move_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask
        move_log_prob[py, px] = -np.inf

        ## Next step probability surface
        next_step_log = step_log_prob.detach().cpu().numpy()[:,:] * x_mask * y_mask
        next_step_log[py, px] = -np.inf

        # -------------------------------------------------------------------------
        # Plot the RGB image, slope, habitat selection, and movement density
        #   Change the panels to visualize different layers
        # -------------------------------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot RGB
        im1 = axs[0, 0].imshow(rgb_image)
        axs[0, 0].set_title('Sentinel-2 RGB')

        # Plot slope
        im2 = axs[0, 1].imshow(slope_subset.detach().cpu().numpy(), cmap='viridis')
        axs[0, 1].set_title('Slope')
        fig.colorbar(im2, ax=axs[0, 1], shrink=0.7)

        # Plot habitat selection
        im3 = axs[1, 0].imshow(hab_log_prob, cmap='viridis')
        axs[1, 0].set_title('Habitat selection log-probability')
        fig.colorbar(im3, ax=axs[1, 0], shrink=0.7)

        # # Movement density (change the axis and uncomment one of the other panels)
        # im3 = axs[1, 0].imshow(move_log_prob, cmap='viridis')
        # axs[1, 0].set_title('Movement log-probability')
        # fig.colorbar(im3, ax=axs[0, 1], shrink=0.7)

        # Next-step probability
        im4 = axs[1, 1].imshow(next_step_log, cmap='viridis')
        axs[1, 1].set_title('Next-step log-probability')
        fig.colorbar(im4, ax=axs[1, 1], shrink=0.7)

        # Save the figure
        filename_covs = f'{output_dir}/sim_S2_id{buffalo_id}_{today_date}_{i}.png'
        plt.tight_layout()
        plt.savefig(filename_covs, dpi=150, bbox_inches='tight')
        # plt.show(block=False)
        plt.close()  # Close the figure to free memory



# %% [markdown]
# ## Plot the simulated trajectory

# %%
# Create a figure and axis with matplotlib
# fig, ax = plt.subplots(figsize=(7.5, 7.5))

# plot RGB from the sentinel layers as the background
month_index = day_to_month_index(starting_yday)
selected_month = f'2019_{month_index:02d}'
# Get the normalized data for the selected month
sentinel_layers = data_dict[selected_month]

# pull out the RGB bands
r_band = sentinel_layers[3,:,:]
g_band = sentinel_layers[2,:,:]
b_band = sentinel_layers[1,:,:]

# Stack the bands along a new axis
rgb_image = np.stack([r_band, g_band, b_band], axis=0)
# Normalize to the range [0, 1] for display
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

# Apply gamma correction to the image
gamma = 1.75
rgb_image = rgb_image ** (1/gamma)

# Plot the raster
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

show(rgb_image, transform=raster_transform, ax=ax)

# Set the title and labels
ax.set_title('Sentinel 2 RGB Image with Simulated Trajectory')
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')

# Number of simulated locations (to just get valid points)
n_sim_points = np.min([x[x>0].shape[0], y[y<0].shape[0]])
print(n_sim_points)

# Plot the simulated trajectory
plt.plot(x[1:n_sim_points], y[1:n_sim_points], color = 'red')
plt.plot(x[1:n_sim_points], y[1:n_sim_points], color = 'red')
plt.show(block=False)


# %% [markdown]
# # Plot with interactive map
# 
# We can also use the `folium` library to plot the trajectory on an interactive map. This is useful for checking the trajectory in more detail, as we can zoom in and out and pan around the map.
# 
# We used the ESRI World Imagery basemap, although other basemaps can be used (check the `folium` documentation for more options: https://python-visualization.github.io/folium/latest/user_guide/raster_layers/tiles.html).
# 
# ### Change the projection of the trajectory
# 
# The `folium` maps use the World Geodetic System 1984 (WGS84) that used latitude and longitute coordinates, denoted by EPSG:4326 as the projection. We will therefore need to reproject the observed data to that projection if we want to plot it.
# 
# We use the `pyproj` library to reproject the data. 

# %%
# Create the reprojection function (input CRS: EPSG 3112, output CRS: EPSG 4326)
coord_transformer = Transformer.from_crs('epsg:3112', 'epsg:4326')

# Observed data
# Convert the easting and northing coordinates to geographic coordinates
buffalo_lon, buffalo_lat = coord_transformer.transform(buffalo_df['x1_'], buffalo_df['y1_'])
print(buffalo_lon[0:2], buffalo_lat[0:2])
# Create a list of coordinate pairs (rather than separate lists)
buffalo_coordinates = [[lon, lat] for lon, lat in zip(buffalo_lon, buffalo_lat)]
print(buffalo_coordinates[0:2])

# Simulated data
# Convert the easting and northing coordinates to geographic coordinates
sim_lon, sim_lat = coord_transformer.transform(x[x>0], y[x>0])
# Create a list of coordinate pairs
sim_coordinates = [[lon, lat] for lon, lat in zip(sim_lon, sim_lat)]

# %% [markdown]
# ## Create the basemap

# %%
# ESRI World Imagery basemap tile
tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

# Create a folium map with the ESRI World Imagery basemap (centred on starting location)
basemap = folium.Map(location=buffalo_coordinates[0], tiles=tiles, attr='Esri', zoom_start = 10)

# %% [markdown]
# ## Show the basemap with the trajectories
# 
# Because the simulated trajectories are stochastic they won't line up with the observed trajectory, but plotting the data should reveal general patterns and relationships to environmental features.

# %%
# Add the buffalo data
folium.PolyLine(
    locations=buffalo_coordinates,
    color='red',
    weight=2,
    opacity=1,
    tooltip='Buffalo trajectory'
).add_to(basemap)

# Add the simulated data
folium.PolyLine(
    locations=sim_coordinates,
    color='blue',
    weight=2,
    opacity=1,
    tooltip='Simulated trajectory'
).add_to(basemap)

# Plot the map
basemap

# %% [markdown]
# ## Assess temporal dynamics of the simulated trajectory
# 
# Create a dataframe from the simulated data

# %%
trajectory_df = pd.DataFrame({'x1': x[x>0], 
                              'y1': y[x>0], 
                              'hour_t1': hour_t1[x>0], 
                              'yday_t1': yday_t1[x>0], 
                              'bearing_prev': bearing[x>0]})

trajectory_df

# %% [markdown]
# Calculate step lengths and turning angles

# %%
# np.arctan2(y[i-1] - y[i-2], x[i-1] - x[i-2])

# Next step
trajectory_df['x2'] = trajectory_df['x1'].shift(-1)
trajectory_df['y2'] = trajectory_df['y1'].shift(-1)
trajectory_df['hour_t2'] = trajectory_df['hour_t1'].shift(-1)
trajectory_df['yday_t2'] = trajectory_df['yday_t1'].shift(-1)
# Pad the missing value with a specified value, e.g., 0
# buffalo_df['bearing_tm1'] = buffalo_df['bearing_tm1'].fillna(0)

# Step length
trajectory_df['step_length'] = np.sqrt((trajectory_df['x2'] - trajectory_df['x1'])**2 + 
                                         (trajectory_df['y2'] - trajectory_df['y1'])**2)

# Calculate the step's bearing
trajectory_df['bearing'] = np.arctan2(trajectory_df['y2'] - trajectory_df['y1'], 
                                             trajectory_df['x2'] - trajectory_df['x1'])

# Turning angle
trajectory_df['turning_angle'] = trajectory_df['bearing'].diff()

for index in range(1, len(trajectory_df)):
    if trajectory_df.loc[index, 'turning_angle'] > np.pi:
        trajectory_df.loc[index, 'turning_angle'] -= 2 * np.pi
    elif trajectory_df.loc[index, 'turning_angle'] < -np.pi:
        trajectory_df.loc[index, 'turning_angle'] += 2 * np.pi

trajectory_df

# %% [markdown]
# Plot the step lengths in relation to the time of day

# %%
# Create jitter for the x-axis (hour_t1)
jitter_amount = 0.25  # Adjust this value to control the amount of jitter
x_jittered = trajectory_df['hour_t1'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(trajectory_df))

x_jittered_buffalo = buffalo_df['hour_t1'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(buffalo_df))

# Plot with jittered x values
plt.scatter(x_jittered, trajectory_df['step_length'], c='blue', s=1, alpha=0.5)
plt.scatter(x_jittered_buffalo[range(min(len(buffalo_df),len(trajectory_df)))], 
            buffalo_df['sl_'][range(min(len(buffalo_df),len(trajectory_df)))], c='red', s=1, alpha=0.5)
plt.xlabel('Hour of the day')
plt.ylabel('Step length (m)')
plt.title('Simulated trajectory')
plt.show(block=False)

# Plot with jittered x values
plt.scatter(x_jittered, np.log(trajectory_df['step_length']), c='blue', s=1, alpha=0.5)
plt.scatter(x_jittered_buffalo[range(min(len(buffalo_df),len(trajectory_df)))], 
            np.log(buffalo_df['sl_'][range(min(len(buffalo_df),len(trajectory_df)))]), c='red', s=1, alpha=0.5)
plt.xlabel('Hour of the day')
plt.ylabel('Step length (m - log scale)')
plt.title('Simulated trajectory')
plt.show(block=False)

# %% [markdown]
# Histograms

# %%
# Create jitter for the x-axis (hour_t1)
jitter_amount = 0.25  # Adjust this value to control the amount of jitter
x_jittered = trajectory_df['hour_t1'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(trajectory_df))

x_jittered_buffalo = buffalo_df['hour_t1'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(buffalo_df))

# Plot with jittered x values
plt.hist(trajectory_df['step_length'], bins=100, color='blue', alpha=0.5, label='Simulated steps')
plt.hist(buffalo_df['sl_'][range(min(len(buffalo_df),len(trajectory_df)))], bins=100, color='red', alpha=0.5, label='Observed steps (Buffalo 2005)')
plt.xlabel('Locations')
plt.xlabel('Step length (m)')
plt.legend()
plt.show(block=False)

# Plot with jittered x values
plt.hist(np.log(trajectory_df['step_length']), bins=100, color='blue', alpha=0.5, label='Simulated steps')
plt.hist(np.log(buffalo_df['sl_'][range(min(len(buffalo_df),len(trajectory_df)))]), bins=100, color='red', alpha=0.5, label='Observed steps (Buffalo 2005)')
plt.xlabel('Locations')
plt.xlabel('Step length (m - log scale)')
plt.legend()
plt.show(block=False)

# %% [markdown]
# Plot the turning angles in relation to the time of day

# %%
# Plot with jittered x values
plt.scatter(x_jittered, trajectory_df['turning_angle'], c='blue', s=1, alpha=0.5)
plt.scatter(x_jittered_buffalo[range(min(len(buffalo_df),len(trajectory_df)))], buffalo_df['ta_'][range(min(len(buffalo_df),len(trajectory_df)))], c='red', s=1, alpha=0.5)
plt.xlabel('Hour of the day')
plt.ylabel('Turning angle (radians)')
plt.title('Simulated trajectory')
plt.show(block=False)

# %% [markdown]
# ## Write the trajectory to a csv

# %% [markdown]
# ### Only run this once otherwise it will create duplicates

# %%
# output directory for saving trajectories
output_dir_traj = f'outputs/deepSSF_trajectories/S2/{buffalo_id}'
os.makedirs(output_dir_traj, exist_ok=True)

# Combine vectors into a DataFrame
trajectory_df = pd.DataFrame({'x': x[x>0], 
                              'y': y[x>0], 
                              'hour': hour_t1[x>0], 
                              'yday': yday_t1[x>0], 
                              'bearing': bearing[x>0]})

n_steps_actual = x[x>0].shape[0]

# Save the DataFrame to a CSV file
index = 1
csv_filename = f'{output_dir_traj}/deepSSF_S2_id{buffalo_id}_{n_steps_actual}steps_{index}_{today_date}.csv'

# Check if the file already exists and find a new name if necessary
while os.path.exists(csv_filename):
    csv_filename = f'{output_dir_traj}/deepSSF_S2_id{buffalo_id}_{n_steps_actual}steps_{index}_{today_date}.csv'
    index += 1

print(csv_filename)
trajectory_df.to_csv(csv_filename, index=True)

# %% [markdown]
# ## Multiple trajectories in a loop

# %%
# -------------------------------------------------------------------------
# Setup parameters
# -------------------------------------------------------------------------
n_trajectories = 1000
n_steps = 3000
starting_yday = 206

# -------------------------------------------------------------------------
# Looping over simulated individuals
# -------------------------------------------------------------------------

for j in range(1, n_trajectories+1):

    # Empty lists to store the x and y coordinates
    x = np.repeat(0., n_steps)
    y = np.repeat(0., n_steps)

    # Set the first entry as the starting location
    x[0], y[0] = start_x, start_y

    # Create sequence of steps
    step = range(1, n_steps)

    # Create an hour-of-day sequence and repeat it until it reaches n_steps.
    hour_t1 = np.resize(range(24), n_steps)

    # Convert hour-of-day values into sine and cosine components.
    hour_t1_sin = np.sin(2 * np.pi * hour_t1 / 24)
    hour_t1_cos = np.cos(2 * np.pi * hour_t1 / 24)

    # Create the day of the year sequences 
    # We want to index the NDVI layers into next year, which is why the ydays go above 365
    yday_t1 = np.repeat(range(starting_yday, starting_yday + 365), 24)
    yday_t1 = np.resize(yday_t1, n_steps)

    # Convert day-of-year values into sine and cosine components.
    yday_t1_sin = np.sin(2 * np.pi * yday_t1 / 365)
    yday_t1_cos = np.cos(2 * np.pi * yday_t1 / 365)

    # Initialise a bearing vector with zeroes for all simulation steps, 
    # which will be updated during the simulation.
    bearing = np.repeat(0., n_steps).astype(np.float32)

    # Convert lists to PyTorch tensors
    hour_t1_tensor = torch.tensor(hour_t1).float()
    hour_t1_sin_tensor = torch.tensor(hour_t1_sin).float()
    hour_t1_cos_tensor = torch.tensor(hour_t1_cos).float()
    yday_t1_tensor = torch.tensor(yday_t1).float()
    yday_t1_sin_tensor = torch.tensor(yday_t1_sin).float()
    yday_t1_cos_tensor = torch.tensor(yday_t1_cos).float()  
    bearing_tensor = torch.tensor(bearing).float()

    # Stack tensors column-wise to create a tensor of shape [n_steps, 4]
    x2_full = torch.stack((hour_t1_sin_tensor, 
                           hour_t1_cos_tensor, 
                           yday_t1_sin_tensor, 
                           yday_t1_cos_tensor), dim=1)

    # Initialize variables to cache the previous yday and month index
    previous_yday = None


    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    
    for i in range(1, n_steps):

        x_loc = x[i-1]
        y_loc = y[i-1]

        # calculate the bearing from the previous location
        if i > 1:
            bearing_rad = np.arctan2(y[i-1] - y[i-2], x[i-1] - x[i-2])
        else:
            # if it's the first step, sample a random bearing
            bearing_rad = np.random.uniform(-np.pi, np.pi)

        # Store the bearing in the vector
        bearing[i-1] = bearing_rad
        # print("Bearing[i-1]", bearing[i-1]) # Debugging print

        # Convert the bearing to a tensor and add dimensions for the batch and channel
        bearing_tensor = torch.tensor(bearing[i-1]).unsqueeze(0).unsqueeze(0)
        # print(bearing_tensor.shape) # Debugging print

        # Select the temporal features for the specific step
        x2 = x2_full[i-1,:].unsqueeze(dim=0)
        # print(x2) # Debugging print

        # Determine the month index based on the day of the year
        day_of_year = yday_t1[i-1]  % 365
        if day_of_year != previous_yday:
            month_index = day_to_month_index(day_of_year)
            previous_yday = day_of_year

        # print(f'Day of the year: {day_of_year}') # Debugging print
        # print(f'Month index: {month_index}') # Debugging print

        # Landscape rasters besides the Sentinel-2 bands
        landscape_raster_list = [slope_landscape_norm]
        
        sim_outputs = simulate_next_step(sentinel_data_dict=data_dict,
                                        which_month=month_index,
                                        landscape_raster_tensors=landscape_raster_list,
                                        scalars_to_grid=x2,
                                        bearing=bearing_tensor,
                                        window_size=window_size,
                                        x_loc=x_loc,
                                        y_loc=y_loc,
                                        landscape_raster_transform=landscape_raster_transform)
        
        (new_x, new_y, 
        hab_log_prob, move_log_prob, step_log_prob, 
        px, py, 
        s2_b2, s2_b3, s2_b4,
        slope_subset) = sim_outputs
        # print(f'New location in pixel coordinates           {px, py}') # Debugging print
        # print(f'New location in geographic coordinates      {new_x, new_y}\n') # Debugging print

        x[i] = new_x
        y[i] = new_y


    # -------------------------------------------------------------------------
    # Save the simulated trajectories
    # -------------------------------------------------------------------------

    # save the data frames individually
    # Combine vectors into a DataFrame
    trajectory_df = pd.DataFrame({'x': x[x>0], 
                                  'y': y[x>0], 
                                  'hour': hour_t1[x>0], 
                                  'yday': yday_t1[x>0], 
                                  'bearing': bearing[x>0]})
    
    n_steps_actual = x[x>0].shape[0]

    # Save the DataFrame to a CSV file
    index = j
    csv_filename = f'{output_dir_traj}/deepSSF_S2_id{buffalo_id}_{n_steps_actual}steps_{index}_{today_date}.csv'

    # Check if the file already exists and find a new name if necessary
    while os.path.exists(csv_filename):
        csv_filename = f'{output_dir_traj}/deepSSF_S2_id{buffalo_id}_{n_steps_actual}steps_{index}_{today_date}.csv'
        index += 1

    print(csv_filename)
    trajectory_df.to_csv(csv_filename, index=True)



