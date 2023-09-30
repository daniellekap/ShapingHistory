#credit: https://github.com/tonystevenj/vae-celeba-pytorch-lightning

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import pytorch_lightning as pl
import torchvision
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 19, 14)

class VAE(pl.LightningModule):
    def __init__(self, image_channels=1, h_dim=19*14*256, z_dim=32, lr = 1e-3):
        self.lr = lr
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=10, stride=1, padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=10, stride=1, padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2, padding = 2, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
#             nn.Tanh(),
#             nn.LeakyReLU(),
        )
      
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(std.device)  # Move eps to the same device as std
        z = mu + std * eps
        return z


    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     # return torch.normal(mu, std)
    #     esp = torch.randn(*mu.size())
    #     z = mu + std * esp
    #     return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        mu = mu.to(h.device)  # Move mu to the same device as h
        logvar = logvar.to(h.device)  # Move logvar to the same device as h
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
#         print("x:",x.shape)
        h = self.encoder(x)
#         print("h:",h.shape)
        
        z, mu, logvar = self.bottleneck(h)
        
        z = self.fc3(z)
        #print("z:",z.shape)
        
        return [self.decoder(z), mu, logvar]
    
# ********************************************************************************************************************************
    
    def loss_fn(self, recon_x, x, mu, logvar):
#         print(recon_x)
#         print(x)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD
    
    
    def loss_function(self,recons,x,mu,logvar):
        # Account for the minibatch samples from the dataset; M_N = self.params['batch_size']/ self.num_train_imgs
        kld_weight = 0.5
        
#         recons_loss =F.mse_loss(recons, input)
        recons_loss =F.mse_loss(recons, x,reduction="sum")

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kld_loss = torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
#         optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4)
#         optimizer = torch.optim.SGD(self.parameters(), lr=1e-5,momentum=0.9)
#         optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-2)
        return optimizer
    
    counter=0
    
    def training_step(self, train_batch, batch_idx):
        self.counter+=1
        x,y= train_batch
        x=x.float()
        z, mu, logvar = self(x)
        loss = self.loss_function(z, x,mu, logvar)
#         loss = self.loss_fn(z, x,mu, logvar)
        if self.counter%50 ==0:
            print(loss)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.float()
        z, mu, logvar = self(x)
        loss = self.loss_function(z, x, mu, logvar)
    #     loss = self.loss_fn(z, x, mu, logvar)
        self.log('val_loss', loss, prog_bar=True)
        return loss