import sys
import json
from pathlib import Path
directory = Path(__file__).resolve()
sys.path.append(str(directory.parent.parent))

import chromadb
from model import llm
from embeddings.scripts.embedding_models import embedding_models, ModelName, client
from prompt_and_state import sanitise_business_description_prompt, CleansedQueryOutput, SearchMetadata, SearchOutput, SearchResult

def run_sic_and_section_semantic_search(query, model_name='minilm'):
    model = embedding_models[model_name]
    query_embedding = model.encode([query], normalize_embeddings=True)
    collection = client.get_collection(f"sic_and_section_{model_name}")
    
    results = collection.query(query_embeddings=query_embedding,n_results=5)
    return results

def run_sic_only_semantic_search(query, model_name='minilm'):
    model = embedding_models[model_name]
    collection = client.get_collection(f"sic_only_{model_name}")
    query_embedding = model.encode([query], normalize_embeddings=True)

    results = collection.query(query_embeddings=query_embedding, n_results=5)

    return results

def clean_query_using_llm(query):
    prompt = sanitise_business_description_prompt(query=query)
    response = llm.with_structured_output(CleansedQueryOutput). invoke(prompt)
    
    return response.cleansed_query
   
    # response = llm.invoke(prompt)

    # return response.content

def calculate_score(chroma_score, score_type):
    if score_type == 'L2_Distance':
        return chroma_score
    else: 
        return 1 - 0.5 * (chroma_score ** 2)
    
def compose_output(original_query, cleansed_query, results, model_name, search_metric, search_type):
    structured_results=[]
    
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        
        sim_score = calculate_score(chroma_score=score, score_type=search_metric)
        structured_results.append(
            SearchResult(
            similarily_score=sim_score,
            sic_code=meta['SIC Code'],
            sic_code_description=doc,
            section_name=meta.get('Section Name', ''),
            section_description=meta.get('Section Description', '')
            )
        )
    meatadata = SearchMetadata(embedding_model_used=model_name, search_metric=search_metric, search_type=search_type)

    output = SearchOutput(search_query=original_query, search_metadata=meatadata, cleansed_query=cleansed_query.strip(), results=structured_results)

    return output

def search(strategy=None, query="", model_name='minilm'):
    if model_name not in embedding_models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(embedding_models.keys())}")
    
    sanitised_query = clean_query_using_llm(query)
    print(f"Cleansed query: {sanitised_query}")
    print(f"Using model: {model_name}")

    if strategy == "sic_and_section":
        results = run_sic_and_section_semantic_search(sanitised_query, model_name)
        return compose_output(results=results, model_name=model_name, original_query=query, cleansed_query=sanitised_query,search_metric='Cosine',search_type='Semantic')
    elif strategy == "sic_and_section_and_keyword":
        return compose_output(results=results, model_name=model_name, original_query=query, cleansed_query=sanitised_query,search_metric='Cosine',search_type='Keyword')
    else:
        results = run_sic_only_semantic_search(sanitised_query, model_name)
        return compose_output(results=results, model_name=model_name, original_query=query, cleansed_query=sanitised_query,search_metric='Cosine',search_type='Semantic')

query = "Octopus is a leading renewable energy supplier based in the UK, known for offering green electricity and gas to households and businesses. The company emphasizes sustainability by sourcing 100'%' of its electricity from renewable sources such as wind, solar, and hydro."
output = search(strategy="sic_only", query=query, model_name=ModelName.MINILM.value)

print(json.dumps(output.model_dump(), indent=2))