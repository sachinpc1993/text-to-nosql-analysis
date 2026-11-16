import argparse
import json
from pprint import pprint

from litellm_call import call_llm

from mongo_client import (
    get_database_schema,
    schema_to_text,
    execute_pipeline,
)

from llm_output_utils import (
    clean_llm_output
)


SYSTEM_PROMPT = """
Write only the MongoDB query for the given natural language question using the schema below. Do not include any explanations or comments. Do not select fields that are not explicitly requested.

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

def run_query(
    database_name: str,
    question: str,
    model: str,
    sample_size: int = 100,
):

    # ---------------------------------------------------------
    # 1. Infer database schema
    # ---------------------------------------------------------

    schema = get_database_schema(
        database_name=database_name,
        sample_size=sample_size,
    )

    schema_text = schema_to_text(schema)

    # ---------------------------------------------------------
    # 2. Generate MongoDB query
    # ---------------------------------------------------------

    generated_query = generate_mongodb_query(
        database_name=database_name,
        question=question,
        model=model,
        schema_text=schema_text,
    )

    collection = generated_query["collection"]
    pipeline = generated_query["pipeline"]

    # ---------------------------------------------------------
    # 3. Validate collection
    # ---------------------------------------------------------

    if collection not in schema:
        raise ValueError(
            f"LLM generated unknown collection: {collection}\n"
            f"Available collections: {list(schema.keys())}"
        )

    # ---------------------------------------------------------
    # 4. Execute MongoDB pipeline
    # ---------------------------------------------------------

    execution = execute_pipeline(
        database_name=database_name,
        collection_name=collection,
        pipeline=pipeline,
    )

    return {
        "question": question,
        "database": database_name,
        "model": model,
        "collection": collection,
        "pipeline": pipeline,
        "execution": execution,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Natural Language → MongoDB runner"
    )

    parser.add_argument(
        "database",
        type=str,
        help="MongoDB database name",
    )

    parser.add_argument(
        "question",
        type=str,
        help="Natural-language question",
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
        help="Documents sampled per collection for schema inference",
    )

    args = parser.parse_args()

    print("\nDatabase")
    print("--------")
    print(args.database)

    print("\nNatural Language Query")
    print("----------------------")
    print(args.question)

    result = run_query(
        database_name=args.database,
        question=args.question,
        model=args.model,
        sample_size=args.sample_size,
    )

    print("\nGenerated MongoDB Query")
    print("-----------------------")
    print(f"Database   : {result['database']}")
    print(f"Collection : {result['collection']}")

    print("\nPipeline:")
    pprint(result["pipeline"])

    print("\nExecution")
    print("---------")

    execution = result["execution"]

    if execution["success"]:

        results = execution["results"]

        print(
            f"Success. Returned {len(results)} document(s).\n"
        )

        for document in results:
            pprint(document)

    else:

        print("MongoDB execution failed.")

        print(
            f"Error type    : {execution['error_type']}"
        )

        print(
            f"Error message : {execution['error_message']}"
        )


if __name__ == "__main__":
    main()