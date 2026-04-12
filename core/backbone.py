import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SharedBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SharedBackbone, self).__init__()
        # pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # input dimension of the original FC layer (512 for ResNet18)
        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()
        
        # MLP projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.projection(features)