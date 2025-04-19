import uuid
import json
import numpy as np
import pandas as pd
import chromadb
from chromadb import Client
from chromadb.config import Settings
from download_sic_codes_data import fetch_sic_codes_df
from embedding_models import embedding_models, collections
from utils import DataPaths

year = 2007
sic_desc_weight, section_desc_weight = 0.8, 0.2


def create_embeddings():
    # Load your cleaned SIC dataframe
    df = fetch_sic_codes_df(year=year)

    # Extract text fields
    sic_descriptions = df["Description"].astype(str).tolist()
    section_descriptions = df["Section Description"].astype(str).tolist()

    # Generate embeddings for each model
    embeddings_data = {}
    for model_name, model in embedding_models.items():
        print(f"Generating embeddings with {model_name}...")

        desc_embeddings = model.encode(
            sic_descriptions, normalize_embeddings=True, show_progress_bar=True
        )
        section_embeddings = model.encode(
            section_descriptions, normalize_embeddings=True, show_progress_bar=True
        )

        # Normalize + weight embeddings
        desc_norm = desc_embeddings / np.linalg.norm(
            desc_embeddings, axis=1, keepdims=True
        )
        section_norm = section_embeddings / np.linalg.norm(
            section_embeddings, axis=1, keepdims=True
        )
        final_embeddings = (
            sic_desc_weight * desc_norm + section_desc_weight * section_norm
        )

        # Save to disk using path enums
        np.save(
            str(DataPaths.SIC_AND_SECTION_EMBEDDINGS.value).format(
                model_name=model_name
            ),
            final_embeddings,
        )
        np.save(
            str(DataPaths.SIC_ONLY_EMBEDDINGS.value).format(model_name=model_name),
            desc_norm,
        )

        embeddings_data[model_name] = {
            "sic_only": desc_norm,
            "sic_and_section": final_embeddings,
        }

    # Generate metadata
    metadata = (
        df[["SIC Code", "Description", "Section Name", "Section Description"]]
        .astype(str)
        .to_dict(orient="records")
    )
    ids = [str(uuid.uuid4()) for _ in metadata]

    with open(DataPaths.METADATA.value, "w") as f:
        for i, meta in zip(ids, metadata):
            json.dump({"id": i, "metadata": meta}, f)
            f.write("\n")

    print("✅ Embeddings saved for all models.")
    return embeddings_data, metadata, ids


def load_embeddings_into_chroma_db(embeddings_data, metadata, ids):
    documents = [record["Description"] for record in metadata]

    for model_name in embedding_models.keys():
        print(f"Loading {model_name} embeddings into ChromaDB...")

        # Load into sic_only collection
        collections[model_name]["sic_only"].add(
            documents=documents,
            embeddings=embeddings_data[model_name]["sic_only"],
            metadatas=metadata,
            ids=ids,
        )

        # Load into sic_and_section collection
        collections[model_name]["sic_and_section"].add(
            documents=documents,
            embeddings=embeddings_data[model_name]["sic_and_section"],
            metadatas=metadata,
            ids=ids,
        )

    print("✅ All collections created in ChromaDB.")


if __name__ == "__main__":
    embeddings_data, metadata, ids = create_embeddings()
    load_embeddings_into_chroma_db(embeddings_data, metadata, ids)
