import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, C, H, W]
            targets: Tensor of shape [B, H, W]
        """
        num_classes = logits.shape[1]
        
        valid_mask = (targets >= 0) & (targets < num_classes)
        
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        targets = targets * valid_mask.long()
        one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        probs = F.softmax(logits, dim=1)
        
        probs = probs * valid_mask.unsqueeze(1).float()
        one_hot = one_hot * valid_mask.unsqueeze(1).float()
        
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1. - dice.mean()
