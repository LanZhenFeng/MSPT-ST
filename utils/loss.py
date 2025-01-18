import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self, auxiliary_loss_weight=0.):
        super().__init__()
        self.auxiliary_loss_weight = auxiliary_loss_weight
    
    def forward(self, input, target, mask):
        # input [batch, length, height, width, channel] or ([batch, length, channel, height, width], auxiliary_loss)
        # target [batch, length, height, width, channel]
        # mask [1height, width]
        if isinstance(input, tuple):
            input, auxiliary_loss = input
            
        loss = (input - target) ** 2 # [batch, length, height, width, channel]
        loss = loss.mean(dim=(0, 1, 4)) # [height, width]
        loss = (loss * mask).sum() / mask.sum() # [1]

        if auxiliary_loss is None:
            return loss
        else:
            return loss + self.auxiliary_loss_weight * auxiliary_loss

class MaskedMAELoss(nn.Module):
    def __init__(self, auxiliary_loss_weight=0.):
        super().__init__()
        self.auxiliary_loss_weight = auxiliary_loss_weight
    
    def forward(self, input, target, mask):
        # input [batch, length, height, width, channel] or ([batch, length, channel, height, width], auxiliary_loss)
        # target [batch, length, height, width, channel]
        # mask [1height, width]
        if isinstance(input, tuple):
            input, auxiliary_loss = input
            
        loss = torch.abs(input - target) # [batch, length, height, width, channel]
        loss = loss.mean(dim=(0, 1, 4)) # [height, width]
        loss = (loss * mask).sum() / mask.sum() # [1]

        if auxiliary_loss is None:
            return loss
        else:
            return loss + self.auxiliary_loss_weight * auxiliary_loss

class MaskedMSEMAELoss(nn.Module):
    def __init__(self, auxiliary_loss_weight=0.):
        super().__init__()
        self.auxiliary_loss_weight = auxiliary_loss_weight
    
    def forward(self, input, target, mask):
        # input [batch, length, height, width, channel] or ([batch, length, channel, height, width], auxiliary_loss)
        # target [batch, length, height, width, channel]
        # mask [1height, width]
        if isinstance(input, tuple):
            input, auxiliary_loss = input
            
        loss = (input - target) ** 2 # [batch, length, height, width, channel]
        loss_mse = loss.mean(dim=(0, 1, 4)) # [height, width]
        loss_mse = (loss_mse * mask).sum() / mask.sum() # [1]

        loss = torch.abs(input - target) # [batch, length, height, width, channel]
        loss_mae = loss.mean(dim=(0, 1, 4)) # [height, width]
        loss_mae = (loss_mae * mask).sum() / mask.sum() # [1]

        loss = loss_mse + loss_mae

        if auxiliary_loss is None:
            return loss
        else:
            return loss + self.auxiliary_loss_weight * auxiliary_loss
