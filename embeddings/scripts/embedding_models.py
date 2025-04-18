import chromadb
from enum import Enum, auto
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./embeddings/chroma_db")

class ModelName(Enum):
    MINILM = "minilm_l6"
    MPNET = "mpnet"
    
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
embedding_models = {
    ModelName.MINILM.value: SentenceTransformer('all-MiniLM-L6-v2'),
    ModelName.MPNET.value: SentenceTransformer('all-mpnet-base-v2')
}

# Create collections for each model
collections = {
    ModelName.MINILM.value: {
        'sic_only': client.get_or_create_collection(f"sic_only_{ModelName.MINILM.value}"),
        'sic_and_section': client.get_or_create_collection(f"sic_and_section_{ModelName.MINILM.value}")
    },
    ModelName.MPNET.value: {
        'sic_only': client.get_or_create_collection(f"sic_only_{ModelName.MPNET.value}"),
        'sic_and_section': client.get_or_create_collection(f"sic_and_section_{ModelName.MPNET.value}")
    }
}

# existing_collections = [col.name for col in client.list_collections()]

# if SIC_AND_SECTION_COLLECTION_NAME in existing_collections:
#     client.delete_collection(SIC_AND_SECTION_COLLECTION_NAME)

# if SIC_COLLECTION_NAME in existing_collections:
#     client.delete_collection(SIC_COLLECTION_NAME)

