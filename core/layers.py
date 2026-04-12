import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute L2 Euclidean distances
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        
        # Hinge loss: max(d(a,p) - d(a,n) + margin, 0)
        loss = torch.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()  