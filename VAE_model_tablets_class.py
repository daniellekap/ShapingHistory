import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 19, 14)

class VAE(pl.LightningModule):
    def __init__(self, image_channels=1, h_dim=19*14*256, z_dim=32, lr=1e-3, use_classification_loss=True, 
                 num_classes=None, loss_type="standard", class_weights=None, device = None):
        
        super(VAE, self).__init__()
        self.lr = lr
        self.use_classification_loss = use_classification_loss
        
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
        )
      
        # self.loss_type = loss_type
        # if loss_type == "weighted":
        #     self.class_weights = torch.tensor(class_weights).to(device)  # pass class_weights as a list, like [0.1, 0.9]
        #     self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        # elif loss_type == "focal":
        #     self.criterion = FocalLoss()

        self.loss_type = loss_type
        if use_classification_loss:
            if loss_type == "standard":
                self.criterion = nn.CrossEntropyLoss()
            elif loss_type == "weighted":
                # Check if class weights are provided
                if class_weights is None:
                    raise ValueError("For weighted loss, class_weights must be provided.")
                self.class_weights = torch.tensor(class_weights).to(device)
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            elif loss_type == "focal":
                self.criterion = FocalLoss()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        
        if self.use_classification_loss:
            assert num_classes is not None, "num_classes must be provided if use_classification_loss is True."
            self.fc_classify = nn.Sequential(
                nn.Linear(z_dim, num_classes),
                nn.Softmax(dim=1)
            )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(std.device)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        if self.use_classification_loss:
            class_logits = self.fc_classify(z)
            return z, mu, logvar, class_logits
        return z, mu, logvar

    def forward(self, x):
        if self.use_classification_loss:
            z, mu, logvar, class_logits = self.bottleneck(self.encoder(x))
            z = self.fc3(z)
            return [self.decoder(z), mu, logvar, class_logits]
        else:
            z, mu, logvar = self.bottleneck(self.encoder(x))
            z = self.fc3(z)
            return [self.decoder(z), mu, logvar]    
    
    def loss_function(self,recons,x,mu,logvar):
        # Account for the minibatch samples from the dataset; M_N = self.params['batch_size']/ self.num_train_imgs
        kld_weight = 0.5
        
#         recons_loss =F.mse_loss(recons, input)
        recons_loss =F.mse_loss(recons, x,reduction="sum")

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kld_loss = torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss

    def classification_loss(self, logits, labels):
        if self.loss_type == "standard":
            return F.cross_entropy(logits, labels)
        else:  # For both "weighted" and "focal"
            return self.criterion(logits, labels)
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self(x)
        
        recon, mu, logvar = outputs[:3]
        recon_loss = self.loss_function(recon, x, mu, logvar)
        total_loss = recon_loss
        
        if self.use_classification_loss:
            class_logits = outputs[3]
            class_loss = self.classification_loss(class_logits, y)
            total_loss += class_loss
            self.log('train_class_loss', class_loss)
            
        self.log('train_recon_loss', recon_loss)
        self.log('train_total_loss', total_loss)
        return total_loss

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self(x)
        
        recon, mu, logvar = outputs[:3]
        recon_loss = self.loss_function(recon, x, mu, logvar)
        total_loss = recon_loss
        
        if self.use_classification_loss:
            class_logits = outputs[3]
            class_loss = self.classification_loss(class_logits, y)
            total_loss += class_loss
            self.log('val_class_loss', class_loss)
            
        self.log('val_recon_loss', recon_loss)
        self.log('val_total_loss', total_loss)
        return total_loss