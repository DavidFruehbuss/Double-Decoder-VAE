# standard packages
import numpy as np
import matplotlib.pyplot as plt
import os

# pytorch
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

# seed
seed = 7
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def generate(model, x):
  x = x.reshape(1,-1)
  x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var = model.forward(x)
  x_rec_ae = x_rec_ae.cpu().detach().numpy()
  plt.figure(2)
  print(x_rec_ae.shape)
  x = x_rec_ae
  x = x.reshape(28,28)
  plt.imshow(x)
  return

def visualize_manifold(model, grid_size=10):
  '''
  For latent space z_dim=2 this returns a visualisation of the learned manifold
  '''

  values = torch.arange(0.5/grid_size, 1, 1/grid_size)

  percentils = torch.distributions.Normal(0, 1) 
  zs = percentils.icdf(values)

  mesh_grid_x, mesh_grid_y = torch.meshgrid(zs, zs, indexing='ij')
  mesh_grid = torch.stack([mesh_grid_x.flatten(), mesh_grid_y.flatten()], dim=1)
  print(mesh_grid.shape)

  z = mesh_grid.to(device)

  sampled_img = model.linear_(z)
  sampled_img = model.decoder(sampled_img)
  sampled_img = F.softmax(sampled_img, dim=1).unsqueeze(1)

  sampled_img = sampled_img.reshape(-1,28,28).cpu().detach().numpy()

  fig, ax = plt.subplots(grid_size,grid_size,figsize=(20,20))
  for i in range(grid_size**2):
    ax[i//grid_size,i%grid_size].imshow(sampled_img[i])

  return sampled_img

def visualize_reconstructions(model):
  '''
  For latent space z_dim=2 this returns a visualisation of the learned manifold
  '''

  z_1 = torch.arange(0, 10, 1)

  mesh_grid_x, mesh_grid_y = torch.meshgrid(z_1, z_1, indexing='ij')
  mesh_grid = torch.stack([mesh_grid_x.flatten(), mesh_grid_y.flatten()], dim=1)

  z = mesh_grid

  z = torch.nn.functional.one_hot(z, 10).reshape(z.shape[0],-1)
  z = z.type(torch.FloatTensor).to(device)

  sampled_img = model.linear_vae(z)
  sampled_img = model.decoder_VAE(sampled_img)
  sampled_img = F.softmax(sampled_img, dim=1).unsqueeze(1)

  sampled_img = sampled_img.reshape(-1,28,28).cpu().detach().numpy()

  fig, ax = plt.subplots(10,10,figsize=(20,20))
  for i in range(10**2):
    ax[i//10,i%10].imshow(sampled_img[i])

  return sampled_img
