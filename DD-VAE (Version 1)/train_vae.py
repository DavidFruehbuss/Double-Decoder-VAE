from datasets import get_mnist
from models import VAE
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

def train(model, learning_rate, num_epochs, dataloader):
    '''
    trains the model
    '''

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):

      epoch_loss = 0
      epoch_rec_loss = 0
      epoch_reg_loss = 0
      instance_loss = 0

      for i, (batch_images, _) in enumerate(dataloader):

        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # for BCE loss
        # batch_images = batch_images / 2 + 0.5

        # pass batch through the model
        x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var  = model.forward(batch_images)

        loss, rec_loss, reg_loss = model.compute_loss(batch_images, x_rec_vae, x_rec_ae, simplex, z_mean, z_log_var)
        
        epoch_loss += loss
        epoch_rec_loss += rec_loss
        epoch_reg_loss += reg_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        instance_loss = loss

        wandb.log({"instance_loss": instance_loss})

      wandb.log({"epoch": e})
      wandb.log({"epoch_loss": epoch_loss})
      wandb.log({"rec_loss": epoch_rec_loss})
      wandb.log({"reg_loss": epoch_reg_loss})
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

    wandb.init(project="test-project", entity="inspired-minds", name='VAE-testing', config=config)

    model = VAE(model_tpye=args.model_type, z_dim=args.z_dim)
    dataloader = get_mnist('train')

    train(model, args.learning_rate, args.epochs, dataloader)
    visualize_manifold(model, grid_size=10)
