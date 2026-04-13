import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Calculates distance: d(a, p) - d(a, n) + margin
        """
        dist_positive = F.pairwise_distance(anchor, positive, p=2)
        dist_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Max(0, ...) ensures we only penalize "hard" or "semi-hard" triplets
        loss = torch.relu(dist_positive - dist_negative + self.margin)
        return loss.mean()