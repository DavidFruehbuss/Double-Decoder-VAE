# standard packages
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as  data

# torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# Wandb
import wandb

# paths
DATASET_PATH = './data'
CECKPOINT_PATH = './checkpoints'

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


class Encoder(nn.Module):

  '''

    Parameters:
      input_channels: number of input channels
      hidden_dim: base number of hidden units
      z_dim: latent dimension / categories of categorical distribution
      act_fn: activation function used throughout the encoder
    '''

  def __init__(
      self, 
      model_type=None, # 'uniform' 'normal'
      input_channels: int=1, 
      hid_dim: int=128,
      z_dim: int=2,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()
    self.model_type = model_type

    self.encoder = nn.Sequential(
        nn.Linear(784, 512),
        act_fn(),
        nn.Linear(512, 256),
        act_fn(),
    )

    self.z_mean = torch.nn.Linear(256,z_dim)
    self.z_log_var = torch.nn.Linear(256,z_dim)

  def forward(self, x):
    x = self.encoder(x)
    z_mean = self.z_mean(x)
    z_log_var = self.z_log_var(x)
    return z_mean, z_log_var

class Decoder(nn.Module):

  '''
  Parameters:
    input_channels: number of input channels
    hidden_dim: base number of hidden units
    z_dim: latent dimension / categories of categorical distribution
    act_fn: activation function used throughout the decoder
  '''

  def __init__(
      self,
      model_type=None, # VAE, AE, uniform, normal
      output_channels: int=1, 
      hid_dim: int=128,
      z_dim: int=2,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()

    self.model_type = model_type

    self.decoder = nn.Sequential(
        nn.Linear(256, 512),
        act_fn(),
        nn.Linear(512,784),
        # nn.Sigmoid(),
    )

    self.linear_ = nn.Sequential(
      nn.Linear(z_dim, 256),
      act_fn()
    )

  def normal_forward(self, z_mean, z_log_var):
    x = self.reparameterize(z_mean, z_log_var.exp())
    # x = torch.distributions.Normal(z_mean, torch.exp(z_log_var)).rsample()
    return self.linear_(x)

  def reparameterize(self, z_mean, z_var):

    assert not (z_var < 0).any().item(), "The reparameterization trick got a negative std as input. "

    noise = torch.randn(z_mean.size()).to(device)
    z = z_mean + noise * z_var
    return z

  def forward(self, x, z_mean=None, z_log_var=None):

    # Normal VAE
    x = self.normal_forward(z_mean, z_log_var)
    x = self.decoder(x)
    return x

class VAE(nn.Module):

  '''

    Parameters:
      batch_size: batch_size
      lr = learning rate
      encoder: encoder module
      decoder: corresponding decoder models
      input_channels: number of input channels
      hidden_dim: base number of hidden units
      z_dim: latent dimension / categories of categorical distribution
    '''

  def __init__(
      self,
      # model type
      model_tpye: str = 'DD-VAE', # 'DD-VAE', N-VAE, U-VAE
      # Encoder, Decoder
      encoder: object = Encoder,
      decoder: object = Decoder,
      # model specifications
      input_channels: int=1, 
      hid_dim: int=128,
      z_dim: int=2,

  ):

    super().__init__()
    self.model_tpye = model_tpye
    self.loss = nn.BCELoss(reduction="none") # BCELoss gives nan values

    self.encoder = encoder('vae',input_channels, hid_dim, z_dim).to(device)
    self.decoder_VAE = decoder('normal', input_channels, hid_dim, z_dim).to(device)

  def forward(self, x):
    '''
    Forward pass through the encoder and both decoders
    '''

    z_mean, z_var = self.encoder(x)
    x_rec_vae = self.decoder_VAE(None, z_mean, z_var)

    return x_rec_vae, None, None, z_mean, z_var


  def compute_loss(self, x, x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var):
    ''' 
    Computer 2 rec_losses and reg_loss
    Reg_loss: KL-Divergence between encoder q_enc(z|x) and uniform prior p(z)
    '''

    # rec_loss = self.loss(x+1e-15, x_rec_vae+1e-15)
    rec_loss = F.mse_loss(x, x_rec_vae, reduction="none")
    rec_loss = rec_loss.sum(dim=[1]).mean(dim=[0])
    reg_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp(), dim = 1), dim = 0)

    return rec_loss + reg_loss, rec_loss, reg_loss

  def _KL(P,Q):
    ''' Kl-Divergence between two distributions '''
    eps = 1e-15
    P = P + eps
    Q = Q + eps
    return torch.sum(P*torch.log(P/Q))