import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

class LightningMNISTClassifier(pl.LightningModule):

  def __init__(self, lr: float, dropout_prob: float = 0.1, distributed: bool = False):
    super(LightningMNISTClassifier, self).__init__()
    
    # mnist images are (1, 28, 28) (channels, width, height) 
    self.layer_1 = nn.Linear(28 * 28, 128)
    self.layer_2 = nn.Linear(128, 256)
    self.layer_3 = nn.Linear(256, 10)
    self.drop_1 = nn.Dropout(dropout_prob)
    self.drop_2 = nn.Dropout(dropout_prob)
    self.lr = lr

    self.distributed = distributed

  def forward(self, x):
      batch_size, channels, width, height = x.size()
      
      # (b, 1, 28, 28) -> (b, 1*28*28)
      x = x.view(batch_size, -1)

      # layer 1 (b, 1*28*28) -> (b, 128)
      x = self.layer_1(x)
      x = torch.relu(x)
      x = self.drop_1(x)

      # layer 2 (b, 128) -> (b, 256)
      x = self.layer_2(x)
      x = torch.relu(x)
      x = self.drop_2(x)

      # layer 3 (b, 256) -> (b, 10)
      x = self.layer_3(x)

      # probability distribution over labels
      x = torch.softmax(x, dim=1)

      return x

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=self.distributed)

      return loss

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=self.distributed)

      return loss

  def test_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=self.distributed)

      return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                    'name': 'expo_lr'}
    return [optimizer], [lr_scheduler]
