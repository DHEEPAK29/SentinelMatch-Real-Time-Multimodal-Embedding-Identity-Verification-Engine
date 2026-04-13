import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
from training.trainer import SiameseTrainer
from models.siamese import SiameseNetwork, get_resnet_backbone

def run_training_pipeline(dataset_dvc_hash, config):
    mlflow.set_experiment("Fingerprint-Verification-Production")
    
    # 1. Backbone and Siamese wrapper instantiation
    backbone = get_resnet_backbone(embedding_dim=128)
    model = SiameseNetwork(backbone)
    
    with mlflow.start_run() as run:
        # 2. Distributed training
        trainer = SiameseTrainer(model=model, device_ids=[0, 1])
        metrics = trainer.fit(dataset_dvc_hash)
        mlflow.log_metrics(metrics)
        
        # 3. Signature and artifact preparation
        model.eval()
        sample_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model.backbone(sample_input)
            
        signature = infer_signature(sample_input.numpy(), output.detach().numpy())
        
        # 4. Tracking and Registry push
        mlflow.set_tags({
            "dataset_dvc_hash": dataset_dvc_hash,
            "git_commit": "current_git_hash"
        })
        
        mlflow.pytorch.log_model(
            pytorch_model=model.backbone, 
            artifact_path="backbone",
            signature=signature,
            input_example=sample_input.numpy(),
            registered_model_name="FingerprintBackbone",
            code_paths=["models/siamese.py"]
        )

if __name__ == "__main__":
    run_training_pipeline("dvc_hash_a1b2c3d4", {"lr": 1e-4})