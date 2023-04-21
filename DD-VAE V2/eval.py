from models import DD_VAE
import torch.optim as optim
import torch
import numpy as np

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

def eval(model, dataloader):
    '''
    evaluates the model
    '''

    model = model.to(device)

    tot_rec_loss_vae = 0.
    tot_reg_loss = 0.
    for i, (batch_images, _) in enumerate(dataloader):
        batch_images = batch_images.to(device)
        batch_images = batch_images.reshape(batch_images.shape[0], -1)

        # optimize reconstruction step
        x_rec_vae, x_rec_ae, simplex = model.forward(batch_images)
        rec_loss_vae, rec_loss_ae, reg_loss = model.optimize_reconstruction(batch_images, x_rec_vae, x_rec_ae, simplex, train=False)

        tot_rec_loss_vae += rec_loss_vae
        tot_reg_loss += reg_loss

    instance_loss_val = tot_rec_loss_vae / len(dataloader)
    reg_loss_val = tot_reg_loss / len(dataloader)
    wandb.log({"instance_loss_val": instance_loss_val,
               "reg_loss_val": reg_loss_val,
               "val_ELBO": instance_loss_val + reg_loss_val})

    print(f'instance_loss_val: {instance_loss_val}, reg_loss_val decoder loss: {reg_loss_val}, rec_loss_vae: {instance_loss_val + reg_loss_val}')