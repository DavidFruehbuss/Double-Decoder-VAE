from datasets import get_mnist
from models import DD_VAE
from utils import *

import argparse
import torch.optim as optim

import wandb

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

def train(model, epochs, dataloader):
    '''
    trains the model
    '''

    self = self.to(device)

    for e in range(epochs):

      epoch_rec_vae_loss = 0
      epoch_rec_ae_loss = 0
      epoch_cross_loss = 0

      for i, (batch_images, _) in enumerate(dataloader):

        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # optimize reconstruction step
        x_rec_vae, x_rec_ae  = self.forward(batch_images)
        rec_loss_vae, rec_loss_ae = self.optimize_reconstruction(batch_images, x_rec_vae, x_rec_ae)

        # optimize approximation step
        x_rec_vae, x_rec_ae  = self.dirichlet_sampling(10)
        cross_loss = self.optimize_approximation(x_rec_vae, x_rec_ae)
        
        epoch_rec_vae_loss += rec_loss_vae
        epoch_rec_ae_loss += rec_loss_ae
        epoch_cross_loss += cross_loss

        wandb.log({"instance_loss": rec_loss_vae})
        wandb.log({"det_decoder_loss": rec_loss_ae})
        wandb.log({"cross_loss": cross_loss})

      wandb.log({"epoch": e})
      wandb.log({"epoch_rec_vae_loss": epoch_rec_vae_loss})
      wandb.log({"epoch_rec_ae_loss": epoch_rec_ae_loss})
      wandb.log({"epoch_cross_loss": epoch_cross_loss})

      print(f'Epoch: {e} done, stochastic decoder loss: {epoch_rec_vae_loss}, deterministic decoder loss: {epoch_rec_ae_loss}, approximation loss: {epoch_cross_loss}')

    print('Training complete')