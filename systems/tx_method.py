"""Skeleton for a new Text-to-MongoDB query-generation approach."""

import argparse
import csv
import json
from pathlib import Path

from litellm_call import call_llm
from llm_output_utils import clean_llm_output
from mongo_client import get_database_schema, schema_to_text


SYSTEM_PROMPT = """
Generate a MongoDB aggregation query for the natural-language question.
Return only valid JSON in this format:
{
  "collection": "<collection_name>",
  "pipeline": [
    {"$match": {...}}
  ]
}
"""


def build_prompt(question: str, schema_text: str) -> str:
    """Add the new method's approach-specific context and instructions here."""
    # TODO: Implement the technique this experiment is intended to evaluate.
    return f"""
Schema:
{schema_text}

Natural-language question:
{question}

Generate the MongoDB query.
""".strip()


def generate_mongodb_query(
    question: str,
    schema_text: str,
    model: str,
) -> dict:
    """Call the model and validate its structured MongoDB query response."""
    prompt = build_prompt(
        question=question,
        schema_text=schema_text,
    )

    response = call_llm(
        model=model,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
    )

    cleaned_response = clean_llm_output(response["content"])
    query = json.loads(cleaned_response)

    if not isinstance(query, dict):
        raise ValueError("The model response must be a JSON object.")
    if not isinstance(query.get("collection"), str):
        raise ValueError("The model response must include a collection name.")
    if not isinstance(query.get("pipeline"), list):
        raise ValueError("The model response must include a pipeline list.")

    return query


def query_to_mongodb_string(query: dict) -> str:
    """Format the structured response as a MongoDB aggregation expression."""
    pipeline = json.dumps(
        query["pipeline"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"db.{query['collection']}.aggregate({pipeline})"


def process_csv(
    input_csv: str,
    run_name: str,
    model: str,
    sample_size: int = 100,
) -> None:
    """Run the approach for each row in the benchmark CSV."""
    input_path = Path(input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    output_dir = Path("output") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    errors_path = output_dir / "errors.csv"

    schema_cache: dict[str, str] = {}

    with (
        input_path.open("r", encoding="utf-8-sig", newline="") as infile,
        predictions_path.open("w", encoding="utf-8", newline="") as predictions_file,
        errors_path.open("w", encoding="utf-8", newline="") as errors_file,
    ):
        reader = csv.DictReader(infile)
        required_columns = {"question_id", "db_id", "question", "query"}
        if reader.fieldnames is None:
            raise ValueError("Input CSV does not contain a header.")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing_columns)}"
            )

        predictions_writer = csv.DictWriter(
            predictions_file,
            fieldnames=["question_id", "question", "query", "predicted_query"],
        )
        predictions_writer.writeheader()

        errors_writer = csv.DictWriter(
            errors_file,
            fieldnames=["question_id", "question", "query", "error"],
        )
        errors_writer.writeheader()

        for row in reader:
            question_id = row["question_id"]
            database_name = row["db_id"]
            question = row["question"]
            gold_query = row["query"]

            try:
                # Cache schema per database so each database is sampled once.
                if database_name not in schema_cache:
                    schema = get_database_schema(
                        database_name=database_name,
                        sample_size=sample_size,
                    )
                    schema_cache[database_name] = schema_to_text(schema)

                generated_query = generate_mongodb_query(
                    question=question,
                    schema_text=schema_cache[database_name],
                    model=model,
                )
                predicted_query = query_to_mongodb_string(generated_query)
            except Exception as exc:
                # Keep failed rows visible in predictions and record the reason.
                predicted_query = ""
                errors_writer.writerow({
                    "question_id": question_id,
                    "question": question,
                    "query": gold_query,
                    "error": str(exc),
                })

            predictions_writer.writerow({
                "question_id": question_id,
                "question": question,
                "query": gold_query,
                "predicted_query": predicted_query,
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skeleton for a new Text-to-MongoDB approach."
    )
    parser.add_argument("input_csv", help="Benchmark CSV with question and query columns.")
    parser.add_argument("--run-name", required=True, help="Name used under output/.")
    parser.add_argument(
        "--model",
        default="ollama/llama3.1:8b",
        help="LiteLLM model name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Documents sampled per collection for schema inference.",
    )
    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        run_name=args.run_name,
        model=args.model,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
