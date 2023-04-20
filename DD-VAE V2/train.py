from datasets import get_mnist
from models import DD_VAE
from utils import *
from eval import eval

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(model, epochs, dataloader, val_loader):
    '''
    trains the model
    '''

    model = model.to(device)

    for e in range(epochs):

      epoch_rec_vae_loss = 0
      epoch_rec_ae_loss = 0
      epoch_reg_loss = 0
      epoch_cross_loss = 0

      for i, (batch_images, _) in enumerate(dataloader):

        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # optimize reconstruction step
        x_rec_vae, x_rec_ae, simplex  = model.forward(batch_images)
        rec_loss_vae, rec_loss_ae, reg_loss = model.optimize_reconstruction(batch_images, x_rec_vae, x_rec_ae, simplex, train=True)

        # optimize approximation step
        for _ in range(model.df):
          # optimize approximation step
          x_rec_vae, x_rec_ae  = model.dirichlet_sampling(model.ds)
          cross_loss = model.optimize_approximation(x_rec_vae, x_rec_ae)
        
        epoch_rec_vae_loss += rec_loss_vae
        epoch_rec_ae_loss += rec_loss_ae
        epoch_reg_loss += reg_loss
        epoch_cross_loss += cross_loss

        wandb.log({"instance_loss": rec_loss_vae})
        wandb.log({"det_decoder_loss": rec_loss_ae})
        wandb.log({"reg_loss": reg_loss})
        wandb.log({"cross_loss": cross_loss})

      print(f'Epoch: {e} done, stochastic decoder loss: {epoch_rec_vae_loss}, deterministic decoder loss: {epoch_rec_ae_loss}, approximation loss: {epoch_cross_loss}, KL loss: {epoch_reg_loss}')

      # Evaluation
      eval(model, val_loader)

    print('Training complete')

if __name__ == '__main__':

    config = {
        "model_type": 'DD-VAE',
        "epochs": 100,
        "batch_size": 1024,
        "z_dim": 20,
        "w_r": 0.1,
        "w_a": 1,
        "ds": 1000,
        "df": 1,
        "d_e_v": True,
        "d_c": 0.1,
        }

    wandb.init(project="test-project", entity="inspired-minds", name='dev', config=config)
    print('Hyperparameters: ', wandb.config)

    model = DD_VAE(model_tpye=wandb.config.model_type, z_dim=wandb.config.z_dim, w_r=wandb.config.w_r, w_a=wandb.config.w_a, ds=wandb.config.ds, df=wandb.config.df, decoder_equal_weights=wandb.config.d_e_v, dirichlet_concentration=wandb.config.d_c)
    dataloader = get_mnist('train')
    val_loader = get_mnist('val')

    train(model, wandb.config.epochs, dataloader, val_loader)