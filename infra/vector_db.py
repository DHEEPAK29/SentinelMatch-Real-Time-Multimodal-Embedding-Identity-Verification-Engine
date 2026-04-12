import numpy as np
from pymilvus import MilvusClient

class VectorDBClient:
    def __init__(self, uri="http://localhost:19530", collection_name="user_biometrics"):
        self.client = MilvusClient(uri)
        self.collection_name = collection_name
        
    def search(self, embedding: np.ndarray, k=1):
        """
        Performs k-Nearest Neighbor search in Milvus.
        """
        # Ensure embedding is (d,) list
        data = embedding.tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[data],
            limit=k,
            output_fields=["id"]
        )
        
        # Mapping results to (ID, distance)
        # Note: Milvus returns 'distance' as the similarity metric
        return [(hit['id'], hit['distance']) for hit in results[0]]

    def ping(self):
        """Check if Milvus server is reachable."""
        try:
            return self.client.has_collection(self.collection_name)
        except:
            return False