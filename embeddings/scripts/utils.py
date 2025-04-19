from enum import Enum


class DataPaths(Enum):
    BASE_DIR = "./embeddings/"
    SIC_AND_SECTION_EMBEDDINGS = (
        BASE_DIR + "file_data/sic_and_section_embeddings_{model_name}.npy"
    )
    SIC_ONLY_EMBEDDINGS = BASE_DIR + "file_data/sic_only_embeddings_{model_name}.npy"
    METADATA = BASE_DIR + "file_data/sic_metadata.json"
    CHROMA_DB = BASE_DIR + "chroma_db"

    @classmethod
    def ensure_paths_exist(cls):
        cls.BASE_DIR.value.mkdir(parents=True, exist_ok=True)
