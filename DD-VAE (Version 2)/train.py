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

    model = model.to(device)

    for e in range(epochs):

      epoch_rec_vae_loss = 0
      epoch_rec_ae_loss = 0
      epoch_cross_loss = 0

      for i, (batch_images, _) in enumerate(dataloader):

        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # optimize reconstruction step
        x_rec_vae, x_rec_ae  = model.forward(batch_images)
        rec_loss_vae, rec_loss_ae = model.optimize_reconstruction(batch_images, x_rec_vae, x_rec_ae)

        # optimize approximation step
        x_rec_vae, x_rec_ae  = model.dirichlet_sampling(10)
        cross_loss = model.optimize_approximation(x_rec_vae, x_rec_ae)
        
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_type', default='DD-VAE', type=str,
                        help='Model type to use')
    parser.add_argument('--z_dim', default=2, type=int,
                        help='latent dimension size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train for')

    args = parser.parse_args()

    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "z_dim": args.z_dim,
        }

    wandb.init(project="test-project", entity="inspired-minds", name='dev', config=config)

    model = DD_VAE(model_tpye=args.model_type, z_dim=args.z_dim)
    dataloader = get_mnist('train')

    train(model, args.epochs, dataloader)