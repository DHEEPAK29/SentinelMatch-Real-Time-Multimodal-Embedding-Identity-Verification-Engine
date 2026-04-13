import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class SiameseTrainer:
    def __init__(self, model, device_ids):
        self.model = DDP(model.to(device_ids[0]), device_ids=device_ids)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = TripletLoss(margin=1.0)

    def train_step(self, anchor, pos, neg):
        self.optimizer.zero_grad()
        
        # Forward pass through shared backbone
        emb_a = self.model(anchor)
        emb_p = self.model(pos)
        emb_n = self.model(neg)
        
        # Compute loss and backprop
        loss = self.criterion(emb_a, emb_p, emb_n)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()