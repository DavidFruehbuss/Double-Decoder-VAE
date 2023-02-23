# standard packages
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

  def __init__(
      self, 
      z_dim: int=2,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Linear(784, 512),
        act_fn(),
        nn.Linear(512, 256),
        act_fn(),
    )

  def forward(self, x):
    return self.encoder(x)

class Decoder(nn.Module):

  def __init__(
      self,
      z_dim: int=2,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()

    self.decoder = nn.Sequential(
        nn.Linear(256, 512),
        act_fn(),
        nn.Linear(512,784),
        # nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.decoder(x)
    return x

class DD_VAE(nn.Module):

  '''

    Parameters:
      batch_size: batch_size
      lr = learning rate
      encoder: encoder module
      decoder: corresponding decoder models
      z_dim: latent dimension / categories of categorical distribution
    '''

  def __init__(
      self,
      # model type
      model_tpye: str = 'DD-VAE', # 'DD-VAE', N-VAE
      # Encoder, Decoder
      encoder: object = Encoder,
      decoder: object = Decoder,
      # model specifications
      z_dim: int=2,
      act_fn: object=nn.ReLU
  ):

    super().__init__()
    self.model_tpye = model_tpye
    self.binary_cross_entropy = nn.BCELoss(reduction="none")

    if self.model_tpye == 'DD-VAE':

      self.encoder = encoder(z_dim).to(device)
      self.linear = nn.Linear(256, z_dim*10)

      # VAE Decoder
      self.linear_vae = nn.Sequential(
        nn.Linear(z_dim*10, 256),
        act_fn()
      )
      self.decoder_VAE = decoder(z_dim).to(device)

      # AE Decoder
      self.linear_ae = nn.Sequential(
        nn.Linear(z_dim*10, 256),
        act_fn()
      )
      self.decoder_AE = decoder(z_dim).to(device)
      # we start with the same weights for the two decoders
      self.decoder_AE.decoder.load_state_dict(self.decoder_VAE.decoder.state_dict())

    elif self.model_tpye == 'N-VAE':

      self.encoder = encoder(z_dim).to(device)
      self.z_mean = torch.nn.Linear(256,z_dim)
      self.z_log_var = torch.nn.Linear(256,z_dim)
      self.linear_ = nn.Sequential(
        nn.Linear(z_dim, 256),
        act_fn()
      )
      self.decoder = decoder(z_dim).to(device)


  def forward(self, x):
    '''
    Forward pass through the encoder and both decoders
    '''

    if self.model_tpye == 'DD-VAE':

      x = self.encoder(x)
      simplex = self.linear(x)
      simplex = simplex.reshape(x.shape[0], -1, 10)
      # simplex = F.softmax(simplex.reshape(x.shape[0], -1, 10), dim=2)

      sample = torch.distributions.Categorical(logits=simplex).sample()
      z_vae = torch.nn.functional.one_hot(sample, 10).reshape(sample.shape[0],-1)
      z_vae = z_vae.type(torch.FloatTensor).to(device)
      z_vae = self.linear_vae(z_vae)
      x_rec_vae = self.decoder_VAE(z_vae)
      x_rec_vae = F.sigmoid(x_rec_vae)
      # x_rec_vae = 0


      z_ae = simplex.reshape(simplex.shape[0],-1)
      z_ae = self.linear_ae(z_ae)
      x_rec_ae = self.decoder_AE(z_ae)
      x_rec_ae = F.sigmoid(x_rec_ae)
      
      simplex_prob = F.softmax(simplex.reshape(x.shape[0], -1, 10), dim=2)

      return x_rec_vae, x_rec_ae, simplex_prob, None, None

    elif self.model_tpye == 'N-VAE':

      x = self.encoder(x)

      z_mean = self.z_mean(x)
      z_log_var = self.z_log_var(x)
      z = self.reparameterize(z_mean, z_log_var.exp())
      # x = torch.distributions.Normal(z_mean, torch.exp(z_log_var)).rsample()
      z = self.linear_(z)

      x_rec_vae = self.decoder(z)
      x_rec_vae = F.sigmoid(x_rec_vae)

      return x_rec_vae, None, None, z_mean, z_log_var

  def reparameterize(self, z_mean, z_var):

    assert not (z_var < 0).any().item(), "The reparameterization trick got a negative std as input. "

    noise = torch.randn(z_mean.size()).to(device)
    z = z_mean + noise * z_var
    return z

  def compute_loss(self, x, x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var):
    ''' 
    Computer 2 rec_losses and reg_loss
    Reg_loss: KL-Divergence between encoder q_enc(z|x) and uniform prior p(z)
    '''
    if self.model_tpye == 'DD-VAE':

      rec_loss_vae = F.mse_loss(x, x_rec_vae, reduction="none")
      # rec_loss_vae = self.binary_cross_entropy(x_rec_vae, x)
      rec_loss_vae = rec_loss_vae.sum(dim=[1]).mean(dim=[0])
      # rec_loss_vae = 0
      rec_loss_ae = F.mse_loss(x, x_rec_ae, reduction="none")
      # rec_loss_ae = self.binary_cross_entropy(x_rec_ae, x)
      rec_loss_ae = rec_loss_ae.sum(dim=[1]).mean(dim=[0])
      # rec_loss_ae = 0
      # p_z = torch.distributions.OneHotCategorical(probs=torch.ones_like(simplex) / 10.0)
      # reg_loss = torch.distributions.kl_divergence(simplex, p_z).sum(-1)
      # p_z = torch.full((2,10),0.1).squeeze(1).to(device)
      # reg_loss = self._KL(simplex, p_z)
      reg_loss = 0

      return rec_loss_vae + rec_loss_ae + reg_loss, rec_loss_vae, reg_loss, rec_loss_ae

    elif self.model_tpye == 'N-VAE':

      rec_loss = F.mse_loss(x, x_rec_vae, reduction="none")
      # rec_loss = self.binary_cross_entropy(x_rec_vae, x)
      rec_loss = rec_loss.sum(dim=[1]).mean(dim=[0])
      reg_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp(), dim = 1), dim = 0)

      return rec_loss + reg_loss, rec_loss, reg_loss, None

  def par_loss(self):
    ''' Par_loss: Mean squared loss between q_vae(x|z) and q_ae(x|P) parameters '''

    return F.mse_loss(self.decoder_VAE.decoder[0].parameters(), self.decoder_AE.decoder[0].parameters())

  def _KL(self,P,Q):
    ''' Kl-Divergence between two distributions '''
    eps = 1e-15
    P = P + eps
    Q = Q + eps
    return torch.sum(P*torch.log(P/Q))

    print('Training complete')