import argparse
import json
import csv


def extract_question_db_mapping(input_json, output_csv):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "db_id"]
        )

        writer.writeheader()

        for item in data:
            writer.writerow({
                "question_id": item["question_id"],
                "db_id": item["db_id"]
            })


def main():
    parser = argparse.ArgumentParser(
        description="Extract question_id and db_id from JSON into CSV."
    )

    parser.add_argument(
        "input_json",
        help="Path to input JSON file"
    )

    parser.add_argument(
        "output_csv",
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    extract_question_db_mapping(
        args.input_json,
        args.output_csv
    )


if __name__ == "__main__":
    main()