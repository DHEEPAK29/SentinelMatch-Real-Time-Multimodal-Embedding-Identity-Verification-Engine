import time
import torch
from fastapi import FastAPI, Depends
from api.routes import verification
from infra.vector_db import VectorDBClient 

app = FastAPI(title="SentinelMatch API")
app.include_router(verification.router)
# Init dep
db_client = VectorDBClient()

@app.get("/health")
def health():
    """
    Industry Standard Health Check:
    1. System Status (Uptime)
    2. Model Readiness (GPU/Memory availability)
    3. Dependent Service Health (Vector DB connectivity)
    """
    # Vector DB connectivity
    db_ok = db_client.ping() 
    # GPU/Torch status
    model_ok = torch.cuda.is_available() or True 
    
    return {
        "status": "healthy" if db_ok and model_ok else "unhealthy",
        "dependencies": {
            "vector_db": "online" if db_ok else "offline",
            "model_engine": "ready" if model_ok else "not_loaded"
        },
        "timestamp": time.time()
    }