import torch.nn as nn
from .backbone import SharedBackbone

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        # Single instance of backbone ensures weight-sharing
        self.backbone = SharedBackbone(embedding_dim=embedding_dim)

    def forward(self, input1, input2):
        # Weight-sharing: pass both inputs through the same instance
        embedding1 = self.backbone(input1)
        embedding2 = self.backbone(input2)
        return embedding1, embedding2

    def get_embedding(self, x):
        """Helper for production inference."""
        return self.backbone(x)