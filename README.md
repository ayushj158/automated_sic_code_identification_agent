# automated_sic_code_identification_agent

# SETUP 

```
#create virtual environment
python -m venv venv-sic-agent   

#activate virtual environment
source venv-sic-agent/bin/activate 

#Install dependencies
pip install -r requirements.txt   
```

# Embeddings
## SIC Code Text Embeddings with ChromaDB

This implementation uses text embeddings for Standard Industrial Classification (SIC) codes, leveraging ChromaDB for efficient similarity search.

## Key Features

- **Text Embedding**: Converts SIC code descriptions into high-dimensional vector representations using transformer-based models
- **Vector Normalization**: Implements L2 normalization on embeddings to ensure all vectors have unit length, making them suitable for cosine similarity
- **ChromaDB Integration**: Utilizes ChromaDB as a vector database for storing and retrieving embeddings
- **Cosine Similarity Search**: Employs cosine similarity metric instead of L2 distance for more accurate semantic matching

## Why Cosine Similarity?

Cosine similarity is preferred over L2 distance because:
- It measures the angle between vectors, focusing on directional similarity
- Is invariant to vector magnitude, making it better for text embeddings
- Performs better for high-dimensional sparse vectors typical in text embeddings

## Usage Notes

- Embeddings are normalized to unit vectors before storage
- ChromaDB collection maintains both embeddings and metadata
- Search queries return semantically similar SIC codes based on description similarity
- Optimal for finding related industry classifications and categories

## CREATE SIC CODE EMBEDDINGS
```
# Run this command and it will auto create all embeddings
python embeddings/scripts/create_siccodes_embe_in_chroma_db.py
```

## Embeeding Models and Types
* Model: MINILM = "minilm_l6"
* Model: MPNET = "mpnet"
* Embedding : SIC Code Description only 
* Embedding : SIC Code Description (*.8) and Section Description(.2) by weights



