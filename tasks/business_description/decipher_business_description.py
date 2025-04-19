import json, math, sys
from pathlib import Path

directory = Path(__file__).resolve()
sys.path.append(str(directory.parent.parent.parent))

from tasks.business_description.bd_prompt_and_state import (
    cleanse_business_description_prompt,
    CleansedQueryOutput,
    SICPredictionOutput,
)
from search.search_on_sic_embeddings import semantic_search
from embeddings.scripts.embedding_models import EmbeddingType, ModelName
from model import llm

minimum_threshold = 0.5
primary_search_strategy = EmbeddingType.SIC_ONLY.value
secondary_search_strategy = EmbeddingType.SIC_AND_SECTION.value


def clean_query_using_llm(query):
    prompt = cleanse_business_description_prompt(query=query)
    response = llm.with_structured_output(CleansedQueryOutput).invoke(prompt)

    return response.cleansed_query

    # response = llm.invoke(prompt)
    # return response.content


def perform_sic_search(
    search_strategy, cleansed_query, model_name=ModelName.MINILM.value, top_k=4
):

    search_output = semantic_search(
        strategy=search_strategy,
        query=cleansed_query,
        model_name=model_name,
        top_k=top_k,
    )

    return search_output.results

def is_results_qualitative(candidates):
        return any(candidate.score >= minimum_threshold for candidate in candidates)

def evaluated_bd(business_description, top_k):
    """
    Evaluates a business description to find the most relevant SIC codes using a two-step search strategy.
    This function performs semantic search using two different strategies (primary and secondary) to find
    relevant SIC codes for a given business description. It first attempts a primary search, and if the
    results are not qualitative enough, proceeds with a secondary search. If needed, it merges results
    from both searches.
    Args:
        business_description (str): The raw business description text to be evaluated
        top_k (int): The number of top candidates to return
    Returns:
        SICPredictionOutput: An object containing:
            - search_query: Original business description
            - cleansed_query: Cleaned version of business description
            - candidates: List of candidate SIC codes with their scores
            - search_vector: Description of search strategy used
            - final_score_ranked: Boolean indicating if results are ranked by final score
    Flow:
        1. Cleanses the input business description
        2. Performs primary search
        3. If primary results are qualitative, returns those
        4. Otherwise, performs secondary search
        5. If secondary results are qualitative, returns those
        6. If neither is satisfactory, merges results (60% primary, 40% secondary)
    """
    cleansed_query = clean_query_using_llm(business_description)

    # Primary search
    primary_candidates = perform_sic_search(
        search_strategy=primary_search_strategy,
        cleansed_query=cleansed_query,
        model_name=ModelName.MINILM.value,
        top_k=top_k,
    )
    primary_candidates.sort(key=lambda x: x.score, reverse=True)

    if is_results_qualitative(primary_candidates):
        return SICPredictionOutput(
            search_query=business_description,
            cleansed_query=cleansed_query,
            candidates=primary_candidates,
            search_vector=f"{primary_search_strategy}_description",
            final_score_ranked=True,
        )

    # Secondary search
    secondary_candidates = perform_sic_search(
        search_strategy=secondary_search_strategy,
        cleansed_query=cleansed_query,
        model_name=ModelName.MINILM.value,
        top_k=top_k,
    )
    
    secondary_candidates.sort(key=lambda x: x.score, reverse=True)
    if is_results_qualitative(secondary_candidates):
        return SICPredictionOutput(
            search_query=business_description,
            cleansed_query=cleansed_query,
            candidates=secondary_candidates,
            search_vector=f"{secondary_search_strategy}_description",
            final_score_ranked=True,
        )

    # Merge results
    primary_count = math.ceil(0.6 * top_k)
    final_candidates = primary_candidates[0:primary_count] + secondary_candidates[0:top_k - primary_count]
    
    return SICPredictionOutput(
        search_query=business_description,
        cleansed_query=cleansed_query,
        candidates=final_candidates,
        search_vector=f"{primary_search_strategy}_description AND {secondary_search_strategy}_description",
        final_score_ranked=True,
    )


query = "Octopus is a leading renewable energy supplier based in the UK, known for offering green electricity and gas to households and businesses. The company emphasizes sustainability by sourcing 100'%' of its electricity from renewable sources such as wind, solar, and hydro."
output = evaluated_bd(query, 4)

print(json.dumps(output.model_dump(), indent=2))
