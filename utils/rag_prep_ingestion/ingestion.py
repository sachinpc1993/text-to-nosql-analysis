import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss_index(
    input_csv: str,
    output_dir: str,
):
    input_path = Path(input_csv)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------
    df = pd.read_csv(input_path)

    required_columns = {
        "question_id",
        "question",
        "query",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}"
        )

    # Remove rows with missing questions
    df = df.dropna(subset=["question"]).reset_index(drop=True)

    questions = df["question"].astype(str).tolist()

    print(f"Loaded {len(questions)} training examples.")

    # ---------------------------------------------------------
    # 2. Load embedding model
    # ---------------------------------------------------------
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # 3. Embed natural-language questions
    # ---------------------------------------------------------
    print("Generating embeddings...")

    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    # ---------------------------------------------------------
    # 4. Build FAISS index
    #
    # Because embeddings are normalized, inner product
    # corresponds to cosine similarity.
    # ---------------------------------------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    # ---------------------------------------------------------
    # 5. Save FAISS index
    # ---------------------------------------------------------
    faiss_path = output_path / "train.index"

    faiss.write_index(
        index,
        str(faiss_path),
    )

    # ---------------------------------------------------------
    # 6. Save metadata
    #
    # FAISS itself only stores vectors.
    # The position in this metadata list corresponds exactly
    # to the FAISS vector ID.
    # ---------------------------------------------------------
    metadata = []

    for faiss_id, row in df.iterrows():
        metadata.append(
            {
                "faiss_id": int(faiss_id),
                "question_id": int(row["question_id"]),
                "question": str(row["question"]),
                "query": str(row["query"]),
            }
        )

    metadata_path = output_path / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # 7. Save embedding model information
    # ---------------------------------------------------------
    config = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(dimension),
        "num_examples": int(len(metadata)),
        "similarity": "cosine",
        "faiss_index_type": "IndexFlatIP",
    }

    config_path = output_path / "config.json"

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print("Index creation complete.")
    print(f"FAISS index : {faiss_path}")
    print(f"Metadata    : {metadata_path}")
    print(f"Config      : {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embed Text-to-NoSQL training questions and "
            "build a local FAISS index."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to train.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="faiss_index",
        help="Directory where the FAISS index and metadata are stored.",
    )

    args = parser.parse_args()

    build_faiss_index(
        input_csv=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()