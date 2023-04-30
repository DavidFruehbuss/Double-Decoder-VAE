from datasets import get_mnist
from models import DD_VAE
from utils import *
from eval import eval

import argparse
import torch.optim as optim
import numpy as np

import wandb

# # seed
# seed = 7
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # ensure reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_type', default='DD-VAE', type=str,
                        help='Model type to use')
    parser.add_argument('--z_dim', default=10, type=int,
                        help='latent dimension size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--w_a', default=0.0, type=float,
                        help='weight of det reconstruction loss')
    parser.add_argument('--w_r', default=0.0, type=float,
                        help='weight of KL loss')
    parser.add_argument('--ds', default=1000, type=int,
                        help='number of dirichlet samples per step')
    parser.add_argument('--df', default=1, type=int,
                        help='number of dirichlet steps per reconstruction step')
    parser.add_argument('--d_e_v', default=True, type=bool,
                        help='if true decoder weights are initalized the same')
    parser.add_argument('--d_c', default=0.1, type=float,
                        help='dirichlet concentration for latent exploration')

    args = parser.parse_args()

    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "z_dim": args.z_dim,
        "w_r": args.w_r,
        "w_a": args.w_a,
        "ds": args.ds,
        "df": args.df,
        "d_e_v": args.d_e_v,
        "dirichlet_concentration": args.d_c
        }

    wandb.init(project="test-project", entity="inspired-minds", name='dev', config=config)

    model = DD_VAE(model_tpye=args.model_type, z_dim=args.z_dim, w_r=args.w_r, w_a=args.w_a, ds=args.ds, df=args.df, decoder_equal_weights=args.d_e_v, dirichlet_concentration=args.d_c)
    dataloader = get_mnist('train')
    val_loader = get_mnist('val')

    train(model, args.epochs, dataloader, val_loader)