import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam


class EraClassifier(pl.LightningModule):
    
    def __init__(self, num_classes=3, LR=1e-3):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.gray_to_triple = nn.Conv2d(1, 3, (1, 1))
        self.core = models.resnet50(pretrained=True)
        
        num_ftrs = self.core.fc.in_features
        
        self.core.fc = nn.Linear(num_ftrs, num_classes)
        
        self.objective = nn.CrossEntropyLoss()
        
        self.LR = LR
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        
    def forward(self, x):
        
        # expected shape for x: (b, 512, 512)
        
        x = x.unsqueeze(1) # (b, 1, 512, 512)
        x = self.gray_to_triple(x) # (b, 3, 512, 512)
        x = self.core(x) # (b, num_classes)
        
        return x
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # Get model predictions
        loss = self.objective(logits, y)  # Compute loss
        
        acc = self.train_acc(logits, y)  
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.objective(logits, y)
        
        acc = self.val_acc(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

class SimpleCNN(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32*32*256, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Applying convolutions, batch norm, activation, and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output for the fully connected layer
        x = x.view(-1, 32*32*256)
        
        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), dtype=torch.float32)
        
        # Log training loss and accuracy
        self.log('train_loss', loss)
        self.log('train_acc', acc)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), dtype=torch.float32)
        
        # Log validation loss and accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
