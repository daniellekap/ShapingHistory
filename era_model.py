import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy
import torchmetrics

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
        logits = self(x)
        loss = self.objective(logits, y)
        self.log('train_loss', loss)
        
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.objective(logits, y)
        self.log('val_loss', loss)
        
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc)
