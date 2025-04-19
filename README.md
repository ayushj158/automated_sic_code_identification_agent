# Intelligent Automated SIC-Code Identification Agent

## SETUP 

```
#create virtual environment
python -m venv venv-sic-agent   

#activate virtual environment
source venv-sic-agent/bin/activate 

#Install dependencies
pip install -r requirements.txt   
```

## Embeddings
### SIC Code Text Embeddings with ChromaDB

This implementation uses text embeddings for Standard Industrial Classification (SIC) codes, leveraging ChromaDB for efficient similarity search.

### Key Features

- **Text Embedding**: Converts SIC code descriptions into high-dimensional vector representations using transformer-based models
- **Vector Normalization**: Implements L2 normalization on embeddings to ensure all vectors have unit length, making them suitable for cosine similarity
- **ChromaDB Integration**: Utilizes ChromaDB as a vector database for storing and retrieving embeddings
- **Cosine Similarity Search**: Employs cosine similarity metric instead of L2 distance for more accurate semantic matching

### Why Cosine Similarity?

Cosine similarity is preferred over L2 distance because:
- It measures the angle between vectors, focusing on directional similarity
- Is invariant to vector magnitude, making it better for text embeddings
- Performs better for high-dimensional sparse vectors typical in text embeddings

### Usage Notes

- Embeddings are normalized to unit vectors before storage
- ChromaDB collection maintains both embeddings and metadata
- Search queries return semantically similar SIC codes based on description similarity
- Optimal for finding related industry classifications and categories

### CREATE SIC CODE EMBEDDINGS
```
# Run this command and it will auto create all embeddings
python embeddings/scripts/create_siccodes_embe_in_chroma_db.py
```

### Embeeding Models and Types
* Model: MINILM = "minilm_l6"
* Model: MPNET = "mpnet"
* Embedding : SIC Code Description only 
* Embedding : SIC Code Description (*.8) and Section Description(.2) by weights



## Tasks List

### Task 1: Business Description - Semantic Search on Sic Code Embeddings 
- [x] Clean the input business description using LLM
- [x] Performs primary semantic search usng sic_code_description only
- [x] If primary results are not qualitative,then secondary search on combined sic_code_description and section_description
- [x] If neither is satisfactory, merges results (60% primary, 40% secondary)
- [x] Only return results with score > threshold

### Task 2: Trading Name - Semantic Search on Sic Code Embeddings 
- [x] Performs primary semantic search usng sic_code_description only embeddings
- [x] Only return results with score > threshold


