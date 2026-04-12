import torch.nn as nn
from .backbone import SharedBackbone
import time
import torch
import redis
from core.model import SharedBackbone

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

class InferenceWorker:
    def __init__(self, batch_size=32, timeout=0.05):
        self.model = SharedBackbone().eval().cuda()
        self.r = redis.Redis(host='localhost', port=6379)
        self.batch_size = batch_size
        self.timeout = timeout

    def run(self):
        while True:
            job_ids = self._collect_batch()
            if not job_ids:
                continue
            
            # Batch inference
            images = [self._load_image(jid) for jid in job_ids]
            tensors = torch.stack(images).cuda()
            
            with torch.no_grad():
                embeddings = self.model(tensors) # Siamese backbone pass
            
            # Store results back in Redis for the API to poll
            for jid, emb in zip(job_ids, embeddings):
                self.r.set(f"job:{jid}:result", str(emb.tolist()))

    def _collect_batch(self):
        """Adaptive batching: wait for batch_size or timeout."""
        batch = []
        start = time.time()
        while len(batch) < self.batch_size and (time.time() - start) < self.timeout:
            item = self.r.lpop("inference_queue")
            if item:
                batch.append(item.decode())
            time.sleep(0.001)
        return batch