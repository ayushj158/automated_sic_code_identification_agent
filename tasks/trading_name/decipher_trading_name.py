import json, math, sys
from pathlib import Path

directory = Path(__file__).resolve()
sys.path.append(str(directory.parent.parent.parent))

from tasks.trading_name.tn_prompt_and_state import (
    SICPredictionOutput,
)
from search.search_on_sic_embeddings import semantic_search
from embeddings.scripts.embedding_models import EmbeddingType, ModelName
from model import llm

minimum_threshold = 0.5
primary_search_strategy = EmbeddingType.SIC_ONLY.value
secondary_search_strategy = EmbeddingType.SIC_AND_SECTION.value


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


def evaluated_trading_name(trading_name, top_k):
    # Primary search
    primary_candidates = perform_sic_search(
        search_strategy=primary_search_strategy,
        cleansed_query=trading_name,
        model_name=ModelName.MINILM.value,
        top_k=top_k,
    )
    filtered_primary_candidates = [candidate for candidate in primary_candidates if candidate.score >=0.3 ]
    filtered_primary_candidates.sort(key=lambda x: x.score, reverse=True)


    return SICPredictionOutput(
        search_query=trading_name,
        cleansed_query=trading_name,
        candidates=filtered_primary_candidates,
        search_vector=f"{primary_search_strategy}_description",
        final_score_ranked=True,
    )

    # if is_results_qualitative(primary_candidates):
    #     return SICPredictionOutput(
    #         search_query=trading_name,
    #         cleansed_query=trading_name,
    #         candidates=primary_candidates,
    #         search_vector=f"{primary_search_strategy}_description",
    #         final_score_ranked=True,
    #     )

    # # Secondary search
    # secondary_candidates = perform_sic_search(
    #     search_strategy=secondary_search_strategy,
    #     cleansed_query = trading_name,
    #     model_name=ModelName.MINILM.value,
    #     top_k=top_k,
    # )

    # secondary_candidates.sort(key=lambda x: x.score, reverse=True)

    # if is_results_qualitative(secondary_candidates):
    #     return SICPredictionOutput(
    #         search_query=trading_name,
    #         cleansed_query=trading_name,
    #         candidates=secondary_candidates,
    #         search_vector=f"{secondary_search_strategy}_description",
    #         final_score_ranked=True,
    #     )

    # # Merge results
    # primary_count = math.ceil(0.6 * top_k)
    # final_candidates = primary_candidates[0:primary_count] + secondary_candidates[0:top_k - primary_count]

    # return SICPredictionOutput(
    #     search_query=trading_name,
    #     cleansed_query=trading_name,
    #     candidates=final_candidates,
    #     search_vector=f"{primary_search_strategy}_description AND {secondary_search_strategy}_description",
    #     final_score_ranked=True,
    # )


query = "Hair Anatomy"
output = evaluated_trading_name(query, 4)

print(json.dumps(output.model_dump(), indent=2))
