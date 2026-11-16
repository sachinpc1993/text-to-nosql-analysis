# Reflect + Repair Baseline

import argparse
import csv
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from pymongo import MongoClient

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

MAX_ATTEMPTS = 3
EXECUTION_SAMPLE_SIZE = 5

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/",
)


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """
Write only the MongoDB query for the given natural language question
using the schema below.

Do not include any explanations or comments.
Do not select fields that are not explicitly requested.

Output Format:
{
  "collection": "<collection_name>",
  "pipeline": [
    {"$match": {...}}
  ]
}
"""


JUDGEMENT_SYSTEM_PROMPT = """
You are evaluating whether the RESULTS returned by an executed MongoDB
query answer a natural-language question.

Base your judgement primarily on the sample data returned after executing
the generated MongoDB query.

Your task is to determine whether the execution sample is relevant to and
appears to answer the natural-language question.

Evaluate the sample by considering:

1. Whether the returned records contain the information requested by the
   question.
2. Whether the returned values appear consistent with the constraints in
   the question, such as requested entities, conditions, ranges, counts,
   grouping, ordering, or aggregation.
3. Whether the shape of the returned results is appropriate for the
   requested answer.
4. Whether requested fields are present in the returned sample.
5. Whether the sample contains clearly irrelevant or contradictory data.

Do NOT judge the query merely by inspecting its MongoDB syntax or pipeline
structure. The judgement must be grounded in the executed result sample.

The execution sample may be incomplete because only a small number of
records are shown. Do not reject a result solely because the sample does
not prove properties that require inspecting the full result set, such as
global ordering or the total number of returned documents.

If the execution sample is empty, judge whether the empty result is
plausibly a correct answer to the question. If there is insufficient
evidence to conclude that the result answers the question, mark it as not
relevant and briefly explain why.

Do not require fields that the question did not request.

Return only valid JSON using exactly this format:

{
  "relevant": true,
  "reason": "<brief reason based on the execution sample>"
}

or

{
  "relevant": false,
  "reason": "<brief description of what is missing or inconsistent in the execution sample>"
}
"""


REPAIR_SYSTEM_PROMPT = """
You are repairing a MongoDB aggregation query.

The previous query either:

1. produced an execution/runtime error, or
2. executed successfully but was judged not to correctly answer the
   natural-language question.

Use the schema, natural-language question, previous MongoDB query,
and the supplied feedback to repair the query.

You may modify only the incorrect part of the query or replace the
entire query if necessary.

Do not include explanations or comments.
Do not select fields that are not explicitly requested.

Return only the repaired MongoDB query.

Output Format:
{
  "collection": "<collection_name>",
  "pipeline": [
    {"$match": {...}}
  ]
}
"""


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def print_llm_metadata(
    llm_response: dict,
    step_name: str,
):
    """
    Print LiteLLM metadata for debugging/evaluation.
    """

    print(f"\n=== LLM METADATA: {step_name} ===")
    print(
        f"Requested model : "
        f"{llm_response.get('requested_model')}"
    )
    print(
        f"Response model  : "
        f"{llm_response.get('response_model')}"
    )
    print(
        f"Provider        : "
        f"{llm_response.get('provider')}"
    )
    print(
        f"Response ID     : "
        f"{llm_response.get('response_id')}"
    )
    print(
        f"Usage           : "
        f"{llm_response.get('usage')}"
    )
    print("=" * 50)


def parse_query_response(
    response: str,
) -> dict:
    """
    Parse and validate an LLM-generated MongoDB query.
    """

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


def json_safe(
    value: Any,
) -> Any:
    """
    Convert MongoDB/Python values into JSON-serialisable values.

    This prevents values such as ObjectId from breaking prompts.
    """

    return json.loads(
        json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------

def generate_mongodb_query(
    database_name: str,
    question: str,
    model: str,
    schema_text: str,
) -> dict:
    """
    Initial Text-to-NoSQL generation.
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
        system_prompt=GENERATION_SYSTEM_PROMPT,
        temperature=0.0,
    )

    print_llm_metadata(
        llm_response,
        "INITIAL GENERATION",
    )

    return parse_query_response(
        llm_response["content"]
    )


# ---------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------

def execute_generated_query(
    database_name: str,
    query: dict,
    sample_size: int = EXECUTION_SAMPLE_SIZE,
) -> dict:
    """
    Execute a generated MongoDB aggregation pipeline.

    Returns:

    {
        "success": True/False,
        "sample": [...],
        "error": None/"..."
    }

    Execution errors are deliberately returned rather than raised so
    that Reflect-and-Repair can use them for the next attempt.
    """

    client = None

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10000,
        )

        database = client[database_name]

        collection_name = query["collection"]
        pipeline = query["pipeline"]

        collection = database[collection_name]

        cursor = collection.aggregate(
            pipeline,
            allowDiskUse=True,
        )

        sample = []

        for document in cursor:
            sample.append(
                json_safe(document)
            )

            if len(sample) >= sample_size:
                break

        return {
            "success": True,
            "sample": sample,
            "error": None,
        }

    except Exception as exc:

        return {
            "success": False,
            "sample": [],
            "error": str(exc),
        }

    finally:
        if client is not None:
            client.close()


# ---------------------------------------------------------------------
# Relevance judgement
# ---------------------------------------------------------------------

def judge_query_relevance(
    database_name: str,
    question: str,
    schema_text: str,
    generated_query: dict,
    execution_sample: list,
    model: str,
) -> dict:
    """
    Judge whether the executed result sample answers the NL question.

    The judgement is grounded in the sample data returned by execution.
    The gold query is NOT supplied to the judge.
    """

    execution_sample_text = json.dumps(
        execution_sample,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    prompt = f"""

Natural Language Question:
{question}

Sample Data:
{execution_sample_text}

A sample of the data is provided. Does the data answer the user's the natural-language question?
"""

    llm_response = call_llm(
        model=model,
        prompt=prompt,
        system_prompt=JUDGEMENT_SYSTEM_PROMPT,
        temperature=0.0,
    )

    print_llm_metadata(
        llm_response,
        "RELEVANCE JUDGEMENT",
    )

    response = clean_llm_output(
        llm_response["content"]
    )

    try:
        judgement = json.loads(response)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Relevance judge returned invalid JSON.\n\n"
            f"Response:\n{response}"
        ) from exc

    if "relevant" not in judgement:
        raise ValueError(
            "Relevance judgement does not contain "
            "'relevant'."
        )

    if "reason" not in judgement:
        raise ValueError(
            "Relevance judgement does not contain "
            "'reason'."
        )

    if not isinstance(
        judgement["relevant"],
        bool,
    ):
        raise ValueError(
            "'relevant' must be true or false."
        )

    return judgement


# ---------------------------------------------------------------------
# Query repair
# ---------------------------------------------------------------------

def repair_mongodb_query(
    database_name: str,
    question: str,
    model: str,
    schema_text: str,
    previous_query: dict,
    feedback: str,
    execution_sample: list | None = None,
    execution_error: str | None = None,
) -> dict:
    """
    Repair the previous MongoDB query.

    The repair can be based on either:
        - semantic/relevance feedback, or
        - an execution/runtime error.
    """

    previous_query_text = json.dumps(
        previous_query,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    if execution_error:
        execution_context = f"""
Execution Status:
FAILED

Execution Error:
{execution_error}
"""

    else:
        sample_text = json.dumps(
            execution_sample or [],
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        execution_context = f"""
Execution Status:
SUCCESS

Sample of Executed Query Result:
{sample_text}
"""

    prompt = f"""
Database:
{database_name}

Schema:
{schema_text}

Natural Language Question:
{question}

Previous MongoDB Query:
{previous_query_text}

{execution_context}

Reflection / Feedback:
{feedback}

Repair the MongoDB query so that it correctly answers the
natural-language question.
"""

    llm_response = call_llm(
        model=model,
        prompt=prompt,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        temperature=0.0,
    )

    print_llm_metadata(
        llm_response,
        "QUERY REPAIR",
    )

    return parse_query_response(
        llm_response["content"]
    )


# ---------------------------------------------------------------------
# Query formatting
# ---------------------------------------------------------------------

def query_to_mongodb_string(
    query: dict,
) -> str:
    """
    Convert structured query to MongoDB shell syntax.
    """

    collection = query["collection"]
    pipeline = query["pipeline"]

    pipeline_json = json.dumps(
        pipeline,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"db.{collection}."
        f"aggregate({pipeline_json})"
    )


# ---------------------------------------------------------------------
# Reflect-and-Repair
# ---------------------------------------------------------------------

def reflect_and_repair(
    database_name: str,
    question: str,
    model: str,
    schema_text: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict, dict, list]:
    """
    Reflect-and-Repair loop.

    Maximum attempts includes the initial generation.

    Attempt 1:
        Generate -> Execute -> Judge

    Attempt 2:
        Repair -> Execute -> Judge

    Attempt 3:
        Repair -> Execute -> Judge

    Execution errors also trigger repair.

    Returns:
        final_query
        final_status
        traces
    """

    current_query = generate_mongodb_query(
        database_name=database_name,
        question=question,
        model=model,
        schema_text=schema_text,
    )

    final_status = {
        "attempts": 0,
        "relevant": False,
        "last_feedback": "",
        "execution_success": False,
    }

    traces = []

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        print("\n" + "#" * 70)
        print(
            f"REFLECT-AND-REPAIR ATTEMPT "
            f"{attempt}/{max_attempts}"
        )
        print("#" * 70)

        print("\nCurrent Query:")
        print(
            query_to_mongodb_string(
                current_query
            )
        )

        final_status["attempts"] = attempt

        attempt_trace = {
            "attempt_number": attempt,
            "candidate_query": json_safe(current_query),
            "execution_status": None,
            "execution_error": None,
            "execution_sample": [],
            "judge_decision": None,
            "judge_reason": None,
            "repaired_query": None,
        }

        execution = execute_generated_query(
            database_name=database_name,
            query=current_query,
        )

        if not execution["success"]:

            error_message = execution["error"]

            print("\nExecution failed:")
            print(error_message)

            final_status["execution_success"] = False
            final_status["relevant"] = False
            final_status["last_feedback"] = error_message

            attempt_trace["execution_status"] = "error"
            attempt_trace["execution_error"] = error_message

            if attempt == max_attempts:
                print("\nMaximum attempts reached.")
                traces.append(attempt_trace)
                break

            feedback = (
                "The MongoDB query could not be executed. "
                "Repair the query using the execution error."
            )

            repaired_query = repair_mongodb_query(
                database_name=database_name,
                question=question,
                model=model,
                schema_text=schema_text,
                previous_query=current_query,
                feedback=feedback,
                execution_error=error_message,
            )

            attempt_trace["repaired_query"] = json_safe(
                repaired_query
            )

            traces.append(attempt_trace)
            current_query = repaired_query

            continue

        final_status["execution_success"] = True

        attempt_trace["execution_status"] = "success"
        attempt_trace["execution_sample"] = json_safe(
            execution["sample"]
        )

        print("\nExecution sample:")
        print(
            json.dumps(
                execution["sample"],
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        judgement = judge_query_relevance(
            database_name=database_name,
            question=question,
            schema_text=schema_text,
            generated_query=current_query,
            execution_sample=execution["sample"],
            model=model,
        )

        print("\nRelevance judgement:")
        print(
            json.dumps(
                judgement,
                indent=2,
                ensure_ascii=False,
            )
        )

        final_status["relevant"] = judgement["relevant"]
        final_status["last_feedback"] = judgement["reason"]

        attempt_trace["judge_decision"] = judgement["relevant"]
        attempt_trace["judge_reason"] = judgement["reason"]

        if judgement["relevant"]:

            print(
                "\nQuery judged relevant. "
                "Stopping Reflect-and-Repair."
            )

            traces.append(attempt_trace)
            break

        if attempt == max_attempts:

            print(
                "\nQuery still judged irrelevant, "
                "but maximum attempts reached."
            )

            traces.append(attempt_trace)
            break

        print(
            "\nQuery judged irrelevant. "
            "Repairing..."
        )

        repaired_query = repair_mongodb_query(
            database_name=database_name,
            question=question,
            model=model,
            schema_text=schema_text,
            previous_query=current_query,
            feedback=judgement["reason"],
            execution_sample=execution["sample"],
        )

        attempt_trace["repaired_query"] = json_safe(
            repaired_query
        )

        traces.append(attempt_trace)
        current_query = repaired_query

    return current_query, final_status, traces


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

def get_schema_text(
    database_name: str,
    sample_size: int,
    schema_cache: dict,
) -> str:
    """
    Load and cache database schema.
    """

    if database_name in schema_cache:
        return schema_cache[database_name]

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

    schema_cache[database_name] = schema_text

    return schema_text


# ---------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------

def process_csv(
    input_csv: str,
    run_name: str,
    model: str,
    sample_size: int = 100,
    max_attempts: int = MAX_ATTEMPTS,
):
    """
    Run Reflect-and-Repair Text-to-NoSQL over a CSV.

    Input:
        question_id
        db_id
        question
        query

    predictions.csv:
        question_id
        question
        query
        predicted_query
        attempt_number

    errors.csv:
        question_id
        question
        query
        error
        attempt_number
    """

    input_path = Path(input_csv)

    output_dir = Path("output") / run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "predictions.csv"
    errors_path = output_dir / "errors.csv"
    traces_dir = output_dir / "traces"

    traces_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print(f"Traces      : {traces_dir}")
    print(f"Max attempts: {max_attempts}")
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

        output_fieldnames = [
            "question_id",
            "question",
            "query",
            "predicted_query",
            "attempt_number",
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
            "attempt_number",
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

            predicted_query = ""
            attempt_number = 0
            final_status = {}
            traces = []

            try:

                schema_text = get_schema_text(
                    database_name=database_name,
                    sample_size=sample_size,
                    schema_cache=schema_cache,
                )

                generated_query, status, traces = reflect_and_repair(
                    database_name=database_name,
                    question=question,
                    model=model,
                    schema_text=schema_text,
                    max_attempts=max_attempts,
                )

                predicted_query = query_to_mongodb_string(
                    generated_query
                )

                attempt_number = status["attempts"]

                print("\nFinal Predicted Query:")
                print(predicted_query)

                print("\nReflect-and-Repair status:")
                print(
                    json.dumps(
                        status,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                final_status = status

                trace_payload = {
                    "question_id": question_id,
                    "db_id": database_name,
                    "question": question,
                    "gold_query": gold_query,
                    "model": model,
                    "run_name": run_name,
                    "max_attempts": max_attempts,
                    "attempt_number": attempt_number,
                    "final_predicted_query": predicted_query,
                    "final_status": final_status,
                    "traces": traces,
                }

                trace_path = (
                    traces_dir /
                    f"{question_id}.json"
                )

                with trace_path.open(
                    "w",
                    encoding="utf-8",
                ) as trace_file:
                    json.dump(
                        trace_payload,
                        trace_file,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )

            except Exception as exc:

                error_message = str(exc)

                print(
                    f"\nERROR for "
                    f"question_id={question_id}: "
                    f"{error_message}"
                )

                error_writer.writerow({
                    "question_id": question_id,
                    "question": question,
                    "query": gold_query,
                    "error": error_message,
                    "attempt_number": attempt_number,
                })

                errorfile.flush()

                trace_payload = {
                    "question_id": question_id,
                    "db_id": database_name,
                    "question": question,
                    "gold_query": gold_query,
                    "model": model,
                    "run_name": run_name,
                    "max_attempts": max_attempts,
                    "attempt_number": attempt_number,
                    "final_predicted_query": predicted_query,
                    "final_status": final_status,
                    "fatal_error": error_message,
                    "traces": traces,
                }

                trace_path = (
                    traces_dir /
                    f"{question_id}.json"
                )

                with trace_path.open(
                    "w",
                    encoding="utf-8",
                ) as trace_file:
                    json.dump(
                        trace_payload,
                        trace_file,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )

            writer.writerow({
                "question_id": question_id,
                "question": question,
                "query": gold_query,
                "predicted_query": predicted_query,
                "attempt_number": attempt_number,
            })

            outfile.flush()

    print("\nFinished.")
    print(f"Run name: {run_name}")
    print(f"Predictions written to: {output_path}")
    print(f"Errors written to: {errors_path}")



# ---------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------

def send_email_notification(
    subject: str,
    body: str,
):
    """
    Send an email notification using SMTP configuration from
    environment variables.

    Required environment variables:
        SMTP_HOST
        SMTP_PORT
        SMTP_USERNAME
        SMTP_PASSWORD
        EMAIL_FROM
        EMAIL_TO

    Email failures are caught so they do not affect the experiment run.
    """

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")
    email_to = os.getenv("EMAIL_TO")

    required_values = {
        "SMTP_HOST": smtp_host,
        "SMTP_USERNAME": smtp_username,
        "SMTP_PASSWORD": smtp_password,
        "EMAIL_FROM": email_from,
        "EMAIL_TO": email_to,
    }

    missing = [
        key
        for key, value in required_values.items()
        if not value
    ]

    if missing:
        print(
            "\nEmail notification skipped. "
            "Missing environment variables: "
            + ", ".join(missing)
        )
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_from
    message["To"] = email_to
    message.set_content(body)

    try:
        with smtplib.SMTP_SSL(
            smtp_host,
            smtp_port,
        ) as server:
            server.login(
                smtp_username,
                smtp_password,
            )
            server.send_message(message)

        print(
            f"\nEmail notification sent to: "
            f"{email_to}"
        )

    except Exception as exc:
        print(
            "\nEmail notification failed: "
            f"{exc}"
        )

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Reflect-and-Repair baseline for "
            "Text-to-MongoDB generation."
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
            "Results are stored in "
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

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "Maximum number of generation/"
            "repair attempts per question."
        ),
    )

    args = parser.parse_args()

    try:
        process_csv(
            input_csv=args.input_csv,
            run_name=args.run_name,
            model=args.model,
            sample_size=args.sample_size,
            max_attempts=args.max_attempts,
        )

        send_email_notification(
            subject=(
                "Text-to-NoSQL run COMPLETED: "
                f"{args.run_name}"
            ),
            body=(
                f"Run name: {args.run_name}\n"
                f"Model: {args.model}\n"
                "Status: COMPLETED\n\n"
                "Details:\n"
                f"Results are available in "
                f"output/{args.run_name}/"
            ),
        )

    except Exception as exc:
        send_email_notification(
            subject=(
                "Text-to-NoSQL run FAILED: "
                f"{args.run_name}"
            ),
            body=(
                f"Run name: {args.run_name}\n"
                f"Model: {args.model}\n"
                "Status: FAILED\n\n"
                f"Error:\n{exc}"
            ),
        )

        raise


if __name__ == "__main__":
    main()
