from pydantic import BaseModel, Field
from operator import add
from typing import List, Optional, Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from search.search_states import SICCandidate
from embeddings.scripts.embedding_models import EmbeddingType


class SICPredictionOutput(BaseModel):
    search_query: str = Field(description="search query passed in input")
    cleansed_query: str = Field(
        description="output when input query is cleansed to reduce noise using LLM"
    )
    search_vector: Annotated[
        str, Field(description="embeddings that will be used for this search")
    ] = EmbeddingType.values()

    final_score_ranked: bool = Field(
        description="Indicates if results are ranked by final score", default=True
    )
    candidates: List[SICCandidate] = Field(
        description="List of potential SIC code matches", min_items=1
    )


class CleansedQueryOutput(BaseModel):
    cleansed_query: str = Field(
        description="cleansed query based on prompt and LLM output"
    )


def cleanse_business_description_prompt(query):
    parser = PydanticOutputParser(pydantic_object=CleansedQueryOutput)

    system_message = """You are an expert in business classification and taxonomy. Your goal is to extract the core economic activity from a customer’s company description. 
                    The output should reflect the primary business function using clear, industry-recognizable language — as would be used to classify it in a government or regulatory context like the UK SIC code system.

                        Your job is NOT to summarize or paraphrase — but to interpret the business as it would be classified in a business registry.

                        DO:
                        - Remove company names, brand tone, and irrelevant fluff
                        - Identify the *core business activity*, not product slogans
                        - Condense to 1–2 sentences describing **what the business actually does**
                        - Focus on services offered, production activities, industries served

                        Respond **only in JSON** and follow this format strictly:
    #                     {format_instructions} """

    # system_message = """You are an expert in business classification and taxonomy and Your goal is to Convert the following company description into a concise business activity phrase.

    #                     Your job is NOT to summarize or paraphrase — but to interpret the business as it would be classified in a business registry.

    #                     Respond **only in JSON** and follow this format strictly:
    #                     {format_instructions} """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", "Description: {query}")]
    )

    tempalte = prompt_template.partial(
        format_instructions=parser.get_format_instructions()
    )

    return tempalte.format_messages(query=query)
