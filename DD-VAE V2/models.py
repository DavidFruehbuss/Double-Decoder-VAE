# standard packages
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.distributions.categorical

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
      z_dim: int=10,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Linear(784, 512),
        act_fn(),
        nn.Linear(512, 256),
        act_fn(),
        nn.Linear(256, z_dim*10),
    )

  def forward(self, x):
    return self.encoder(x)

class Decoder(nn.Module):

  def __init__(
      self,
      z_dim: int=10,
      act_fn: object=nn.ReLU
  ):
    
    super().__init__()

    self.decoder = nn.Sequential(
        nn.Linear(z_dim*10, 256),
        act_fn(),
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
      model_tpye: str = 'DD-VAE',
      # Encoder, Decoder
      encoder: object = Encoder,
      decoder: object = Decoder,
      # model specifications
      z_dim: int=2,
      w_r: int=0.1,
      w_a: int=0.01,
      ds: int=10,
      df: int=1,
      decoder_equal_weights: bool=True,
      dirichlet_concentration: float=0.1,
      learning_rate_rec = 1e-3,
      learning_rate_app = 1e-3,
      act_fn: object=nn.ReLU
  ):

    super().__init__()
    self.model_tpye = model_tpye
    self.binary_cross_entropy = nn.BCELoss(reduction="none")
    self.kl_loss = torch.nn.KLDivLoss(reduction="none")
    self.z_dim = z_dim
    self.w_r = w_r
    self.w_a = w_a
    self.ds = ds
    self.df = df

    self.encoder = encoder(z_dim).to(device)

    # stochastic decoder (generative process)
    self.decoder_VAE = decoder(z_dim).to(device)

    # deterministic decoder (approximation)
    self.decoder_AE = decoder(z_dim).to(device)

    if decoder_equal_weights:
      # we start with the same weights for the two decoders
      self.decoder_AE.decoder.load_state_dict(self.decoder_VAE.decoder.state_dict())

    # dirichlet prior
    concentration = torch.ones(10) * dirichlet_concentration
    self.prior = torch.distributions.dirichlet.Dirichlet(concentration)

    # optimizers
    def concat_generators(*args):
      for gen in args:
          yield from gen

    self.optimizer_rec = optim.Adam(concat_generators(self.encoder.encoder.parameters(), self.decoder_VAE.decoder.parameters()), lr=learning_rate_rec)
    self.optimizer_app = optim.Adam(self.parameters(), lr=learning_rate_app)

  def dirichlet_sampling(self, num_samples=1000):
    '''
    sample from dirichlet prior to explore latent space
    '''
    simplex_batch = self.prior.rsample((num_samples, self.z_dim)).detach()
    simplex_batch = simplex_batch.to(device)

    with torch.no_grad():
      # generative process (stochastic decoder)
      sample = Categorical(probs=simplex_batch).sample()
      z_vae = torch.nn.functional.one_hot(sample, 10).reshape(sample.shape[0],-1)
      z_vae = z_vae.type(torch.FloatTensor).to(device)
      x_rec_vae = self.decoder_VAE(z_vae)
      x_rec_vae = F.sigmoid(x_rec_vae)

    # approximation of generative process (determininstic decoder)
    z_ae = simplex_batch.reshape(simplex_batch.shape[0],-1)
    x_rec_ae = self.decoder_AE(z_ae)
    x_rec_ae = F.sigmoid(x_rec_ae)

    return x_rec_vae, x_rec_ae

  def forward(self, x):
    '''
    Forward pass through the encoder and both decoders
    '''

    simplex = self.encoder(x)
    # simplex = simplex.reshape(x.shape[0], -1, 10) # does not work if we use prob in dirichlet
    simplex_S = F.softmax(simplex.reshape(x.shape[0], -1, 10), dim=2) # (careful: gradient bottleneck)

    # generative process (stochastic decoder)
    sample = torch.distributions.Categorical(probs=simplex_S).sample()
    z_vae = torch.nn.functional.one_hot(sample, 10).reshape(sample.shape[0],-1)
    z_vae = z_vae.type(torch.FloatTensor).to(device)
    x_rec_vae = self.decoder_VAE(z_vae)
    x_rec_vae = F.sigmoid(x_rec_vae)

    # approximation of generative process (determininstic decoder)
    z_ae = simplex_S.reshape(simplex_S.shape[0],-1)
    x_rec_ae = self.decoder_AE(z_ae)
    x_rec_ae = F.sigmoid(x_rec_ae)

    return x_rec_vae, x_rec_ae, simplex_S

  def optimize_reconstruction(self, x, x_rec_vae, x_rec_ae, simplex, train):
    ''' 
    Takes output of forward as input
    Computes rec_losses for each decoder
    '''

    rec_loss_vae = self.binary_cross_entropy(x_rec_vae, x)
    rec_loss_vae = rec_loss_vae.sum(1).mean()
    # rec_loss_vae = 0

    rec_loss_ae = self.binary_cross_entropy(x_rec_ae, x)
    rec_loss_ae = rec_loss_ae.sum(1).mean()
    # rec_loss_ae = 0

    p_z = torch.full((self.z_dim,10),0.1).squeeze(1).to(device)
    reg_loss = self.kl_loss(torch.log(simplex), p_z).sum()
    # reg_loss = 0

    reconstruction_loss = rec_loss_vae + self.w_a * rec_loss_ae + self.w_r * reg_loss

    if train:
      reconstruction_loss.backward()
      self.optimizer_rec.step()
      self.optimizer_rec.zero_grad()

    return rec_loss_vae, rec_loss_ae, reg_loss

  def optimize_approximation(self, x_rec_vae, x_rec_ae):
    ''' 
    Takes output of dirchlet sampling as input
    Computer cross_decoder approximation loss and optimize 
    deterministic decoder to approximate stochastic decoder
    '''

    x_rec_vae = x_rec_vae.detach()
    cross_loss = self.binary_cross_entropy(x_rec_ae, x_rec_vae) # ANESI-Loss
    cross_loss = cross_loss.sum(1).mean()

    cross_loss.backward()
    self.optimizer_app.step()
    self.optimizer_app.zero_grad()

    return cross_loss
  