import io
import torch
from PIL import Image
from torchvision.transforms import v2

class ImagePreprocessor:
    def __init__(self, size=(224, 224)):
        """
        Standardizes input for the Siamese backbone.
        Normalization values correspond to standard ImageNet statistics.
        """
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare(self, image_bytes: bytes) -> torch.Tensor:
        """
        Converts raw bytes to a normalized tensor ready for GPU batching.
        Returns: (1, 3, H, W)
        """
        try:
        # 1. Byte-stream conversion
        stream = io.BytesIO(image_bytes)
        
        # 2. Open and validate format
        with Image.open(stream).convert("RGB") as img:
            # 3. Apply transformation pipeline
            # transform already includes Resize, ToTensor, Normalize
            return self.transform(img).unsqueeze(0)
            
        except (UnidentifiedImageError, OSError, ValueError) as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return None so the caller (batcher) can filter this out
            return None