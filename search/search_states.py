
from pydantic import BaseModel, Field
from operator import add
from typing import List, Optional, Literal, Annotated

class SICCandidate(BaseModel):
    score: float = Field(
        description="Similarity score between 0 and 1"
    )
    sic_code: int 
    sic_code_description: str
    section_name: str
    section_description: str
    rationale: str = Field(description="Explanation for why this SIC code was selected", default=None)

class SearchMetadata(BaseModel):
    embedding_model_used: str =  Field(default=None, description="model name name for embeddings")
    search_metric: Literal["Cosine", "L2_Distance"] = Field(default=None, description="search metric used for similarity scoring")
    search_type: Literal["Semantic","Keyword","LLM_Search"] = Field(default=None, description="Type of search performed")

class SearchOutput(BaseModel):
    search_query: str = Field(description="search query")
    search_metadata: SearchMetadata
    results: Annotated[List[SICCandidate], add]