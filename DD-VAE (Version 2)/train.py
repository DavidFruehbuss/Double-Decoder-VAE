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

def train(model, learning_rate, epochs, dataloader):
    '''
    trains the model
    '''

    self = self.to(device)
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    for e in range(epochs):

      epoch_loss = 0
      epoch_rec_loss = 0
      epoch_reg_loss = 0
      instance_loss = 0
      epoch_rec_loss_ae = 0

      for i, (batch_images, _) in enumerate(dataloader):

        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # pass batch through the model
        x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var  = self.forward(batch_images)

        loss, rec_loss, reg_loss, rec_loss_ae = self.compute_loss(batch_images, x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var)
        
        epoch_loss += loss
        epoch_rec_loss += rec_loss
        epoch_reg_loss += reg_loss
        if rec_loss_ae != None:
          epoch_rec_loss_ae += rec_loss_ae

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        instance_loss = loss

        wandb.log({"instance_loss": instance_loss})

      wandb.log({"epoch": e})
      wandb.log({"epoch_loss": epoch_loss})
      wandb.log({"rec_loss": epoch_rec_loss})
      wandb.log({"reg_loss": epoch_reg_loss})
      wandb.log({"rec_loss_ae": epoch_rec_loss_ae})
      print(f'Epoch: {e} done, Loss: {epoch_loss}, Rec_Loss: {epoch_rec_loss}, Reg_Loss: {epoch_reg_loss}')

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
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "z_dim": args.z_dim,
        }

    wandb.init(project="test-project", entity="inspired-minds", name='dev', config=config)

    model = DD_VAE(model_tpye=args.model_type, z_dim=args.z_dim)
    dataloader = get_mnist('train')

    train(model, args.learning_rate, args.epochs, dataloader)

    if args.model_type == 'DD-VAE':
      visualize_reconstructions(model)
    elif args.model_type == 'N-VAE':
      visualize_manifold(model, grid_size=10)
    else:
      print('Unknown model type')
