# Schema + Fields + CoT

import argparse
import csv
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from litellm_call import call_llm

from mongo_client import (
    get_database_schema,
    schema_to_text,
)

from llm_output_utils import (
    clean_llm_output,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_FAISS_DIR = "faiss_train_agg"
DEFAULT_TOP_K = 10


SYSTEM_PROMPT = ""


# ---------------------------------------------------------------------
# FAISS Retriever
# ---------------------------------------------------------------------

class FaissFewShotRetriever:
    """
    Dense retriever over the DocSpider training examples.

    Expected directory structure:

        faiss_train_agg/
            config.json
            metadata.json
            train.index
    """

    def __init__(
        self,
        index_dir: str,
    ):
        self.index_dir = Path(index_dir)

        self.config_path = self.index_dir / "config.json"
        self.metadata_path = self.index_dir / "metadata.json"
        self.index_path = self.index_dir / "train.index"

        self._validate_files()

        self.config = self._load_json(
            self.config_path
        )

        self.metadata = self._load_json(
            self.metadata_path
        )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.embedding_model_name = self.config[
            "embedding_model"
        ]

        self.embedding_dimension = self.config[
            "embedding_dimension"
        ]

        self.similarity = self.config.get(
            "similarity",
            "cosine",
        )

        print("\nLoading retrieval model")
        print("-" * 70)
        print(
            f"Embedding model : "
            f"{self.embedding_model_name}"
        )
        print(
            f"Embedding dim   : "
            f"{self.embedding_dimension}"
        )
        print(
            f"FAISS index     : "
            f"{self.index_path}"
        )
        print(
            f"Examples        : "
            f"{len(self.metadata)}"
        )
        print(
            f"Similarity      : "
            f"{self.similarity}"
        )
        print("-" * 70)

        self.model = SentenceTransformer(
            self.embedding_model_name
        )

        self._validate_index()

    def _validate_files(self):
        """
        Ensure all required FAISS files exist.
        """

        required_files = [
            self.config_path,
            self.metadata_path,
            self.index_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required FAISS file does not exist: {path}"
                )

    @staticmethod
    def _load_json(path: Path):
        """
        Load JSON file.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as infile:
            return json.load(infile)

    def _validate_index(self):
        """
        Validate consistency between the FAISS index,
        configuration, and metadata.
        """

        if self.index.d != self.embedding_dimension:
            raise ValueError(
                "FAISS index dimension does not match "
                "config.json.\n"
                f"Index dimension: {self.index.d}\n"
                f"Config dimension: {self.embedding_dimension}"
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "Number of vectors in FAISS index does not "
                "match metadata.json.\n"
                f"Index vectors: {self.index.ntotal}\n"
                f"Metadata rows: {len(self.metadata)}"
            )

    def encode_question(
        self,
        question: str,
    ) -> np.ndarray:
        """
        Encode a natural-language question.

        For cosine similarity with IndexFlatIP,
        the embedding is L2-normalized.
        """

        embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        if self.similarity.lower() == "cosine":
            faiss.normalize_L2(
                embedding
            )

        return embedding

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Retrieve the top-k semantically similar
        NL question / NoSQL query pairs.

        Returns items containing:

            rank
            score
            faiss_id
            question_id
            question
            query
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        query_embedding = self.encode_question(
            question
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        retrieved_examples = []

        for rank, (
            faiss_id,
            score,
        ) in enumerate(
            zip(
                indices[0],
                scores[0],
            ),
            start=1,
        ):
            if faiss_id < 0:
                continue

            metadata_item = self.metadata[
                int(faiss_id)
            ]

            retrieved_examples.append({
                "rank": rank,
                "score": float(score),
                "faiss_id": int(faiss_id),
                "question_id": metadata_item.get(
                    "question_id"
                ),
                "question": metadata_item[
                    "question"
                ],
                "query": metadata_item[
                    "query"
                ],
            })

        return retrieved_examples


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------
# The prompt structure mirrors the paper template:
# Task -> Schema -> Natural Language Question -> NoSQL Query Examples.

def format_retrieved_examples(
    examples: list[dict],
) -> str:
    """
    Convert retrieved examples into the requested prompt format.
    """

    blocks = []

    for example in examples:
        block = (
            f"Example {example['rank']}.\n"
            f"{example['question']}\n"
            f"{example['query']}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(
    database_name: str,
    question: str,
    schema_text: str,
    retrieved_examples: list[dict],
) -> str:
    """
    Build the exact user prompt passed to the LLM.
    """

    examples_text = format_retrieved_examples(
        retrieved_examples
    )

    prompt = f"""
Task:
Write only the MongoDB query for the given natural language question using the schema and examples below.
Do not include any explanations or comments.
Do not select fields that are not explicitly requested.

Output Format:
{{
  "collection": "<collection_name>",
  "pipeline": [
    {{"$match": {{...}}}}
  ]
}}

Schema:
{schema_text}

Natural Language Question:
{question}

NoSQL Query Examples:

{examples_text}
""".strip()

    return prompt


def generate_mongodb_query(
    model: str,
    prompt: str,
    retrieved_examples: list[dict],
) -> dict:
    """
    Generate a MongoDB aggregation query using:

        schema
        target NL question
        retrieved NL-NoSQL demonstrations
    """

    print("\n=== RETRIEVED FEW-SHOT EXAMPLES ===")

    for example in retrieved_examples:
        print(
            f"\nRank {example['rank']} "
            f"| Score: {example['score']:.6f} "
            f"| Question ID: {example['question_id']}"
        )

        print(
            f"Question: {example['question']}"
        )

        print(
            f"Query: {example['query']}"
        )

    print(
        "\n===================================\n"
    )

    llm_response = call_llm(
        model=model,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
    )

    print("\n=== LLM METADATA ===")
    print(
        f"Requested model : "
        f"{llm_response['requested_model']}"
    )
    print(
        f"Response model  : "
        f"{llm_response['response_model']}"
    )
    print(
        f"Provider        : "
        f"{llm_response['provider']}"
    )
    print(
        f"Response ID     : "
        f"{llm_response['response_id']}"
    )
    print(
        f"Usage           : "
        f"{llm_response['usage']}"
    )
    print("====================\n")

    response = llm_response["content"]

    cleaned_response = clean_llm_output(
        response
    )

    try:
        query = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "The LLM did not return valid JSON.\n\n"
            f"Raw response:\n{response}\n\n"
            f"Cleaned response:\n{cleaned_response}"
        ) from exc

    if "collection" not in query:
        raise ValueError(
            "LLM response does not contain 'collection'."
        )

    if "pipeline" not in query:
        raise ValueError(
            "LLM response does not contain 'pipeline'."
        )

    if not isinstance(
        query["pipeline"],
        list,
    ):
        raise ValueError(
            "'pipeline' must be a list."
        )

    return query


# ---------------------------------------------------------------------
# MongoDB formatting
# ---------------------------------------------------------------------

def query_to_mongodb_string(
    query: dict,
) -> str:
    """
    Convert structured LLM output to MongoDB shell syntax.
    """

    collection = query["collection"]
    pipeline = query["pipeline"]

    pipeline_json = json.dumps(
        pipeline,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"db.{collection}.aggregate("
        f"{pipeline_json})"
    )


# ---------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------

def get_schema_text(
    database_name: str,
    sample_size: int,
    schema_cache: dict,
) -> str:
    """
    Load database schema while caching schemas so that
    each database is sampled only once.
    """

    if database_name in schema_cache:
        return schema_cache[
            database_name
        ]

    print(
        f"\nLoading schema for database: "
        f"{database_name}"
    )

    schema = get_database_schema(
        database_name=database_name,
        sample_size=sample_size,
    )

    schema_text = schema_to_text(
        schema
    )

    schema_cache[
        database_name
    ] = schema_text

    return schema_text


# ---------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------

def process_csv(
    input_csv: str,
    run_name: str,
    model: str,
    faiss_dir: str,
    top_k: int = 10,
    sample_size: int = 100,
):
    """
    Read Text-to-NoSQL examples from CSV and generate
    retrieval-augmented predictions.

    Expected input columns:

        question_id
        db_id
        question
        query

    Results:

        output/<run_name>/predictions.csv
        output/<run_name>/errors.csv
        output/<run_name>/retrieval.csv
        output/<run_name>/prompts/PROMPT_<QUESTION_ID>.txt
    """

    input_path = Path(
        input_csv
    )

    output_dir = (
        Path("output") / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompts_dir = (
        output_dir / "prompts"
    )

    prompts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "predictions.csv"
    )

    errors_path = (
        output_dir / "errors.csv"
    )

    retrieval_path = (
        output_dir / "retrieval.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: "
            f"{input_path}"
        )

    retriever = FaissFewShotRetriever(
        index_dir=faiss_dir
    )

    print("\nOutput configuration")
    print("-" * 70)
    print(
        f"Run name       : {run_name}"
    )
    print(
        f"Output dir     : {output_dir}"
    )
    print(
        f"Predictions    : {output_path}"
    )
    print(
        f"Errors         : {errors_path}"
    )
    print(
        f"Retrieval log  : {retrieval_path}"
    )
    print(
        f"Prompts dir    : {prompts_dir}"
    )
    print(
        f"FAISS dir      : {faiss_dir}"
    )
    print(
        f"Top K          : {top_k}"
    )
    print("-" * 70)

    schema_cache = {}

    with (
        input_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as infile,

        output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as outfile,

        errors_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as errorfile,

        retrieval_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as retrieval_file,
    ):

        reader = csv.DictReader(
            infile
        )

        required_columns = {
            "question_id",
            "db_id",
            "question",
            "query",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Input CSV does not contain a header."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        output_fieldnames = [
            "question_id",
            "question",
            "query",
            "predicted_query",
        ]

        writer = csv.DictWriter(
            outfile,
            fieldnames=output_fieldnames,
        )

        writer.writeheader()

        error_fieldnames = [
            "question_id",
            "question",
            "query",
            "error",
        ]

        error_writer = csv.DictWriter(
            errorfile,
            fieldnames=error_fieldnames,
        )

        error_writer.writeheader()

        retrieval_fieldnames = [
            "question_id",
            "target_question",
            "rank",
            "similarity_score",
            "retrieved_faiss_id",
            "retrieved_question_id",
            "retrieved_question",
            "retrieved_query",
        ]

        retrieval_writer = csv.DictWriter(
            retrieval_file,
            fieldnames=retrieval_fieldnames,
        )

        retrieval_writer.writeheader()

        for row_number, row in enumerate(
            reader,
            start=1,
        ):
            question_id = row[
                "question_id"
            ]

            database_name = row[
                "db_id"
            ]

            question = row[
                "question"
            ]

            gold_query = row[
                "query"
            ]

            print(
                "\n" + "=" * 70
            )

            print(
                f"Row {row_number} | "
                f"Question ID: {question_id}"
            )

            print(
                f"Database: {database_name}"
            )

            print(
                f"Question: {question}"
            )

            print("=" * 70)

            try:
                retrieved_examples = (
                    retriever.retrieve(
                        question=question,
                        top_k=top_k,
                    )
                )

                for example in retrieved_examples:
                    retrieval_writer.writerow({
                        "question_id": question_id,
                        "target_question": question,
                        "rank": example["rank"],
                        "similarity_score": (
                            example["score"]
                        ),
                        "retrieved_faiss_id": (
                            example["faiss_id"]
                        ),
                        "retrieved_question_id": (
                            example[
                                "question_id"
                            ]
                        ),
                        "retrieved_question": (
                            example[
                                "question"
                            ]
                        ),
                        "retrieved_query": (
                            example[
                                "query"
                            ]
                        ),
                    })

                retrieval_file.flush()

                schema_text = get_schema_text(
                    database_name=database_name,
                    sample_size=sample_size,
                    schema_cache=schema_cache,
                )

                llm_prompt = build_prompt(
                    database_name=database_name,
                    question=question,
                    schema_text=schema_text,
                    retrieved_examples=(
                        retrieved_examples
                    ),
                )

                prompt_path = (
                    prompts_dir
                    / f"PROMPT_{question_id}.txt"
                )

                with prompt_path.open(
                    "w",
                    encoding="utf-8",
                ) as prompt_file:
                    prompt_file.write(
                        llm_prompt
                    )

                generated_query = (
                    generate_mongodb_query(
                        model=model,
                        prompt=llm_prompt,
                        retrieved_examples=(
                            retrieved_examples
                        ),
                    )
                )

                predicted_query = (
                    query_to_mongodb_string(
                        generated_query
                    )
                )

                print(
                    "\nPredicted Query:"
                )

                print(
                    predicted_query
                )

            except Exception as exc:
                error_message = str(
                    exc
                )

                print(
                    f"\nERROR for "
                    f"question_id={question_id}: "
                    f"{error_message}"
                )

                predicted_query = ""

                error_writer.writerow({
                    "question_id": question_id,
                    "question": question,
                    "query": gold_query,
                    "error": error_message,
                })

                errorfile.flush()

            writer.writerow({
                "question_id": question_id,
                "question": question,
                "query": gold_query,
                "predicted_query": predicted_query,
            })

            outfile.flush()

    print("\nFinished.")
    print(
        f"Run name: {run_name}"
    )
    print(
        f"Predictions written to: "
        f"{output_path}"
    )
    print(
        f"Errors written to: "
        f"{errors_path}"
    )
    print(
        f"Retrieval results written to: "
        f"{retrieval_path}"
    )
    print(
        f"Prompts written to: "
        f"{prompts_dir}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MongoDB queries using "
            "retrieval-augmented few-shot prompting."
        )
    )

    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help=(
            "Name of the run. Results will be stored "
            "in output/<run-name>/"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama3.1:8b",
        help="LiteLLM model name",
    )

    parser.add_argument(
        "--faiss-dir",
        type=str,
        default=DEFAULT_FAISS_DIR,
        help=(
            "Directory containing config.json, "
            "metadata.json and train.index"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of retrieved few-shot examples. "
            "Use 3 for the 3-shot system or "
            "10 for the 10-shot system."
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Documents sampled per collection "
            "for schema inference."
        ),
    )

    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        run_name=args.run_name,
        model=args.model,
        faiss_dir=args.faiss_dir,
        top_k=args.top_k,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
