
from pydantic import BaseModel, Field
from operator import add
from typing import List, Optional, Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class CleansedQueryOutput(BaseModel):
    cleansed_query: str = Field(description="cleansed query based on prompt and LLM output")

class SearchResult(BaseModel):
    similarily_score: float = Field(description='similarity score of search result')
    sic_code: int 
    sic_code_description: str
    section_name: str
    section_description: str

class SearchMetadata(BaseModel):
    embedding_model_used: str =  Field(default=None, description="model name name for embeddings")
    search_metric: Literal["Cosine", "L2_Distance"] = Field(default=None, description="search metric used for similarity scoring")
    search_type: Literal["Semantic","Keyword","LLM_Search"] = Field(default=None, description="Type of search performed")

class SearchOutput(BaseModel):
    search_query: str = Field(description="search query")
    cleansed_query: str = Field(description="refined query to reduce noise using LLM")
    search_metadata: SearchMetadata
    results: Annotated[List[SearchResult], add]


def sanitise_business_description_prompt(query):
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
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Description: {query}")
    ])
    
    tempalte= prompt_template.partial(format_instructions=parser.get_format_instructions())

    return tempalte.format_messages(query=query)