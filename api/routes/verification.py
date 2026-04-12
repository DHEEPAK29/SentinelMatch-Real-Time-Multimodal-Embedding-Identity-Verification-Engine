import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from core.model import SiameseNetwork
from infra.vector_db import VectorDBClient # client for Milvus/FAISS

router = APIRouter()
# Load model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device).eval()
db = VectorDBClient()

# Standard ResNet transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/verify")
async def verify_identity(file: UploadFile = File(...)):
    try:
        # 1. Preprocess
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)

        # 2. Extract embedding
        with torch.no_grad():
            embedding = model.get_embedding(tensor).cpu().numpy().flatten()

        # 3. Perform k-NN search
        # Query DB for the most similar identity
        result = db.search(embedding, k=1)
        
        if not result:
            raise HTTPException(status_code=404, detail="Identity not found")

        # Thresholding logic for verification
        # Assuming result contains (id, distance/similarity_score)
        match_id, score = result[0]
        threshold = 0.85
        
        return {
            "match": float(score) >= threshold,
            "score": float(score),
            "identity_id": match_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))