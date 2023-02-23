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
    x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var  = model.forward(x)
    x_rec_vae = x_rec_vae.cpu().detach().numpy()
    plt.figure(2)
    x = x_rec_vae[0,0]
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
  # print(mesh_grid.shape)

  z_mean = mesh_grid.to(device)
  z_log_var = torch.zeros(mesh_grid.shape).to(device) - 1e15

  sampled_img = model.decoder_VAE(model, z_mean, z_log_var)
  sampled_img = F.softmax(sampled_img, dim=1).unsqueeze(1)

  sampled_img = sampled_img.reshape(-1,28,28).cpu().detach().numpy()

  fig, ax = plt.subplots(grid_size,grid_size,figsize=(20,20))
  for i in range(grid_size**2):
    ax[i//grid_size,i%grid_size].imshow(sampled_img[i])

  return sampled_img

def sample(model, num_samples=16):
  z_mean = torch.Tensor([[-0.5,-0.2]]*16).unsqueeze(0).to(device)
  z_log_var = torch.Tensor([[0,0]]*16).unsqueeze(0).to(device)
  sampled_img = model.decoder_VAE(model, z_mean, z_log_var).cpu().detach().numpy()
  sampled_img = sampled_img.reshape(num_samples, 28, 28)
  fig, ax = plt.subplots(4,4)
  for i in range(num_samples):
    ax[i//4,i%4].imshow(sampled_img[i])
