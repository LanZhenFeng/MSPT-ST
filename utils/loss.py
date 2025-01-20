import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self, auxiliary_loss_weight=0.):
        super().__init__()
        self.auxiliary_loss_weight = auxiliary_loss_weight
    
    def forward(self, input, target, mask):
        # input [batch, length, height, width, channel] or ([batch, length, channel, height, width], auxiliary_loss)
        # target [batch, length, height, width, channel]
        # mask [height, width]
        if isinstance(input, tuple):
            input, auxiliary_loss = input

        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # [1, 1, height, width, 1]
        input = input * mask
        target = target * mask

        loss = (input - target) ** 2 # [batch, length, height, width, channel]
        loss = loss.mean()

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

        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # [1, 1, height, width, 1]
        input = input * mask
        target = target * mask

        loss = torch.abs(input - target) # [batch, length, height, width, channel]
        loss = loss.mean()
            

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

        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # [1, 1, height, width, 1]
        input = input * mask
        target = target * mask
        
        loss_mse = (input - target) ** 2 # [batch, length, height, width, channel]
        loss_mse = loss_mse.mean()

        loss_mae = torch.abs(input - target) # [batch, length, height, width, channel]
        loss_mae = loss_mae.mean()
        loss = loss_mse + loss_mae

        if auxiliary_loss is None:
            return loss
        else:
            return loss + self.auxiliary_loss_weight * auxiliary_loss
