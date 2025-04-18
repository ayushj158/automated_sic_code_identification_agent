import sys
import json
from pathlib import Path

directory = Path(__file__).resolve()
sys.path.append(str(directory.parent.parent))

import chromadb
from model import llm
from embeddings.scripts.embedding_models import (
    embedding_models,
    ModelName,
    client,
    EmbeddingType,
)
from search.search_states import (
    SearchMetadata,
    SearchOutput,
    SICCandidate,
)

def run_sic_and_section_semantic_search(query, model_name="minilm", top_k=5):
    model = embedding_models[model_name]
    query_embedding = model.encode([query], normalize_embeddings=True)
    collection = client.get_collection(f"sic_and_section_{model_name}")

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results


def run_sic_only_semantic_search(query, model_name="minilm",top_k=5):
    model = embedding_models[model_name]
    collection = client.get_collection(f"sic_only_{model_name}")
    query_embedding = model.encode([query], normalize_embeddings=True)

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return results

def calculate_score(chroma_score, score_type):
    if score_type == "L2_Distance":
        return chroma_score
    else:
        return 1 - 0.5 * (chroma_score**2)

def generate_rationale(score, strategy):
    prefix = ''
    if strategy == EmbeddingType.SIC_ONLY.value:
        prefix = 'with SIC Code Descriptions'
    elif strategy == EmbeddingType.SIC_AND_SECTION.value:
         prefix = 'with combined search on SIC Code Descriptions and Section Descriptions'
    else: 
        prefix = ''

    if score >= 0.75:
        reason = f"High semantic similarity {prefix} with SIC description"
    elif score >= 0.5:
        reason = f"Moderate semantic similarity {prefix}, partial concept match"
    else:
        reason = f"Weak semantic similarity {prefix}, Low Confidence: Review Suggested"
    
    return reason

def compose_output(original_query, results, model_name, 
                   search_metric, search_type, strategy):
    structured_results = []

    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):

        sim_score = calculate_score(chroma_score=score, score_type=search_metric)
        structured_results.append(
            SICCandidate(
                score=sim_score,
                sic_code=meta["SIC Code"],
                sic_code_description=doc,
                section_name=meta.get("Section Name", ""),
                section_description=meta.get("Section Description", ""),
                rationale=generate_rationale(sim_score, strategy)
            )
        )
    meatadata = SearchMetadata(
        embedding_model_used=model_name,
        search_metric=search_metric,
        search_type=search_type,
    )

    output = SearchOutput(
        search_query=original_query,
        search_metadata=meatadata,
        results=structured_results,
    )

    return output

def semantic_search(strategy=EmbeddingType.SIC_ONLY.value, query="", model_name=ModelName.MINILM.value, top_k=5):
    if model_name not in embedding_models:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(embedding_models.keys())}"
        )
    

    if strategy == EmbeddingType.SIC_AND_SECTION.value:
        print(f"Using model: {model_name} and Strategy:{strategy}")
        results = run_sic_and_section_semantic_search(query, model_name, top_k)
        return compose_output(
            results=results,
            model_name=model_name,
            original_query=query,
            search_metric="Cosine",
            search_type="Semantic",
            strategy=strategy
        )
    elif strategy == EmbeddingType.SIC_ONLY.value:
        print(f"Using model: {model_name} and Strategy:{strategy}")
        results = run_sic_only_semantic_search(query, model_name, top_k)
        return compose_output(
            results=results,
            model_name=model_name,
            original_query=query,
            search_metric="Cosine",
            search_type="Semantic",
            strategy=strategy
        )
    else:
        raise ValueError(
            f"Strategy {strategy} not found. Available strategies: {EmbeddingType.values()}"
        )
