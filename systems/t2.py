# Schema + Fields + CoT

import argparse
import csv
import json
from pathlib import Path

from litellm_call import call_llm

from mongo_client import (
    get_database_schema,
    schema_to_text,
)

from llm_output_utils import (
    clean_llm_output,
)


SYSTEM_PROMPT = """
Write only the MongoDB query for the given natural language question using the schema below.
Do not include any explanations or comments.
Do not select fields that are not explicitly requested.
Let's think step by step.

Output Format:
{
  "collection": "<collection_name>",
  "pipeline": [
    {"$match": {...}}
  ]
}
"""


def generate_mongodb_query(
    database_name: str,
    question: str,
    model: str,
    schema_text: str,
) -> dict:
    """
    Generate a MongoDB aggregation query from a natural-language question.
    """

    prompt = f"""
Database:
{database_name}

Schema:
{schema_text}

Natural Language Question:
{question}

Generate the MongoDB query.
"""

    llm_response = call_llm(
        model=model,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
    )

    print("\n=== LLM METADATA ===")
    print(f"Requested model : {llm_response['requested_model']}")
    print(f"Response model  : {llm_response['response_model']}")
    print(f"Provider        : {llm_response['provider']}")
    print(f"Response ID     : {llm_response['response_id']}")
    print(f"Usage           : {llm_response['usage']}")
    print("====================\n")

    response = llm_response["content"]

    cleaned_response = clean_llm_output(response)

    try:
        query = json.loads(cleaned_response)

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

    if not isinstance(query["pipeline"], list):
        raise ValueError(
            "'pipeline' must be a list."
        )

    return query


def query_to_mongodb_string(query: dict) -> str:
    """
    Convert the structured LLM response into a MongoDB shell-style query.

    Example:

    {
        "collection": "singer",
        "pipeline": [
            {"$match": {"Country": "France"}}
        ]
    }

    becomes:

    db.singer.aggregate([
        {"$match": {"Country": "France"}}
    ])
    """

    collection = query["collection"]
    pipeline = query["pipeline"]

    pipeline_json = json.dumps(
        pipeline,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"db.{collection}.aggregate({pipeline_json})"


def get_schema_text(
    database_name: str,
    sample_size: int,
    schema_cache: dict,
) -> str:
    """
    Load a database schema, caching it so that the same database
    is not sampled repeatedly for multiple CSV rows.
    """

    if database_name in schema_cache:
        return schema_cache[database_name]

    print(f"\nLoading schema for database: {database_name}")

    schema = get_database_schema(
        database_name=database_name,
        sample_size=sample_size,
    )

    schema_text = schema_to_text(schema)

    schema_cache[database_name] = schema_text

    return schema_text


def process_csv(
    input_csv: str,
    run_name: str,
    model: str,
    sample_size: int = 100,
):
    """
    Read Text-to-NoSQL examples from a CSV and generate predictions.

    Expected input columns:
        question_id
        db_id
        question
        query

    Results are written to:

        output/<run_name>/predictions.csv
        output/<run_name>/errors.csv

    predictions.csv columns:
        question_id
        question
        query
        predicted_query

    errors.csv columns:
        question_id
        question
        query
        error
    """

    input_path = Path(input_csv)

    # Create:
    # output/<run_name>/
    output_dir = Path("output") / run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "predictions.csv"
    errors_path = output_dir / "errors.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist: {input_path}"
        )

    print("\nOutput configuration")
    print("-" * 70)
    print(f"Run name    : {run_name}")
    print(f"Output dir  : {output_dir}")
    print(f"Predictions : {output_path}")
    print(f"Errors      : {errors_path}")
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
    ):

        reader = csv.DictReader(infile)

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
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # predictions.csv
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

        # errors.csv
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

        for row_number, row in enumerate(
            reader,
            start=1,
        ):
            question_id = row["question_id"]
            database_name = row["db_id"]
            question = row["question"]
            gold_query = row["query"]

            print("\n" + "=" * 70)

            print(
                f"Row {row_number} | "
                f"Question ID: {question_id}"
            )

            print(f"Database: {database_name}")
            print(f"Question: {question}")

            print("=" * 70)

            try:
                schema_text = get_schema_text(
                    database_name=database_name,
                    sample_size=sample_size,
                    schema_cache=schema_cache,
                )

                generated_query = generate_mongodb_query(
                    database_name=database_name,
                    question=question,
                    model=model,
                    schema_text=schema_text,
                )

                predicted_query = query_to_mongodb_string(
                    generated_query
                )

                print("\nPredicted Query:")
                print(predicted_query)

            except Exception as exc:
                error_message = str(exc)

                print(
                    f"\nERROR for question_id={question_id}: "
                    f"{error_message}"
                )

                predicted_query = ""

                # Write failed example to errors.csv
                error_writer.writerow({
                    "question_id": question_id,
                    "question": question,
                    "query": gold_query,
                    "error": error_message,
                })

                # Flush immediately so errors are preserved
                errorfile.flush()

            # Always write the row to predictions.csv.
            # Failed rows have an empty predicted_query.
            writer.writerow({
                "question_id": question_id,
                "question": question,
                "query": gold_query,
                "predicted_query": predicted_query,
            })

            # Flush after every row so progress is not lost
            outfile.flush()

    print("\nFinished.")
    print(f"Run name: {run_name}")
    print(f"Predictions written to: {output_path}")
    print(f"Errors written to: {errors_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MongoDB queries for questions "
            "contained in a CSV file."
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
            "Name of the run. Results will be stored in "
            "output/<run-name>/"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama3.1:8b",
        help="LiteLLM model name",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Documents sampled per collection "
            "for schema inference"
        ),
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