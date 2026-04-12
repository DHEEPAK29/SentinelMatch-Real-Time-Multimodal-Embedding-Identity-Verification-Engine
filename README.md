# SentinelMatch
SentinelMatch is a high-performance, production-grade Siamese Network service designed for real-time identity verification, biometric authentication, and similarity search. It leverages a weight-sharing backbone for latent space projection and integrates with vector databases for sub-millisecond retrieval.

## Core Architecture
* **Backbone:** Shared feature extractor using ResNet-18 to project inputs into a 128-d latent space.
* **Loss Function:** Triplet Margin Loss for contrastive discriminative learning.
* **Inference:** Optimized via ONNX/TensorRT for low-latency production serving.
* **Serving:** FastAPI-based RESTful API with asynchronous worker support.

```
/siamese-vision-platform  
├── api/  
│   ├── openapi.json        # Contract-first definition for human/machine consumers  
│   ├── llms.txt            # AI-agent instructions & endpoint discovery  
│   ├── main.py             # FastAPI entry point  
│   └── routes/             # API endpoint definitions (Verification, Health)  
├── core/  
│   ├── backbone.py         # Shared ResNet-18 feature extractor  
│   ├── layers.py           # Contrastive/Triplet logic  
│   └── model.py            # Siamese wrapper  
├── ingestion/  
│   ├── stream.py           # Kafka/PubSub listener  
│   └── preproc.py          # Real-time inference image transforms  
├── training/  
│   ├── trainer.py          # Distributed training loop  
│   └── loss.py             # Triplet Loss implementation  
├── infra/  
│   ├── Dockerfile          # Container definition  
│   ├── k8s/                # Kubernetes deployment manifests  
│   └── vector_db.py        # FAISS/Milvus client logic  
├── docs/                   # Developer & consumer documentation  
├── tests/                  # Unit and contract tests  
├── config.yaml             # Hyperparameters & Infrastructure endpoints  
├── requirements.txt  
└── README.md
```

## API Serving Perspective

The following components are critical for production API deployment and AI-agent compatibility:

* **`api/`**: Contains the OpenAPI contract (`openapi.json`) and AI-discovery metadata (`llms.txt`). These are mandatory for enabling AI agents to discover, understand, and interact with the system's capabilities autonomously.
* **`infra/vector_db.py`**: Essential for serving logic. The API queries this module directly to perform real-time similarity lookups during inference requests.
* **`config.yaml`**: Houses environment-specific configurations, including model registry URLs, database connection strings, and authentication tokens required for secure API deployment.
* **`ingestion/`**: Manages upstream services. In a decoupled architecture, these populate the feature store that the API queries to maintain up-to-date embedding data.
* **`training/`**: Houses model development and training pipelines. While separate from the serving path, this component produces the artifacts consumed by the API.

## AI-API Convergence Implementation

To ensure seamless integration with AI agents, we utilize two primary standards:

1. **`openapi.json`**: Defines the API schema. AI agents utilize this to programmatically understand capabilities, required arguments, and return types without human intervention.
2. **`llms.txt`**: A lightweight, plain-text summary located in the root or `api/` directory. This file provides agents with high-level guidance on system interaction, operational limitations, and preferred paths for automated tasks.  

## Key Features
* **Real-Time Verification:** Sub-200ms latency for end-to-end identity matching.
* **Scalable Retrieval:** Integration with FAISS/Milvus for k-Nearest Neighbor (k-NN) search.
* **Production-Ready:** Containerized with Docker, health-check endpoints, and model versioning.

## System Pipeline
1. **Ingestion:** Data stream ingestion (Kafka/PubSub).
2. **Embedding:** Real-time feature projection through shared weights.
3. **Serving:** Vector DB lookup for identification or duplicate detection.



## Getting Started
### Prerequisites
- Python 3.10+
- PyTorch 2.x
- Docker & Kubernetes

### Deployment
1. Build the image: `docker build -t sentinel-match:latest .`
2. Configure environment: Set `VECTOR_DB_URL` and `MODEL_PATH` in `config.yaml`.
3. Launch: `docker-compose up`

## License
MIT