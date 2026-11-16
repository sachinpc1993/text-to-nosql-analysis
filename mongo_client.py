import json
from pymongo import MongoClient
from pymongo.errors import PyMongoError


client = MongoClient("mongodb://localhost:27017/")


def mongo_type(value):
    """
    Convert Python values into MongoDB-friendly type names.
    """
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return "str"

    if isinstance(value, dict):
        return "object"

    if isinstance(value, list):
        if not value:
            return "array"

        element_types = {
            mongo_type(item)
            for item in value
        }

        if len(element_types) == 1:
            element_type = next(iter(element_types))
            return f"array<{element_type}>"

        return f"array<{' | '.join(sorted(element_types))}>"

    return type(value).__name__


def extract_fields(
    obj,
    prefix="",
    fields=None
):
    """
    Recursively extract nested MongoDB fields.

    Example:
        ENCOUNTERS
        ENCOUNTERS.DESCRIPTION
        ENCOUNTERS.MEDICATIONS
        ENCOUNTERS.MEDICATIONS.CODE
    """
    if fields is None:
        fields = {}

    if not isinstance(obj, dict):
        return fields

    for key, value in obj.items():

        field_name = f"{prefix}.{key}" if prefix else key

        fields.setdefault(field_name, set())
        fields[field_name].add(mongo_type(value))

        # Nested object
        if isinstance(value, dict):

            extract_fields(
                value,
                prefix=field_name,
                fields=fields
            )

        # Array
        elif isinstance(value, list):

            for item in value:

                # Array of objects
                if isinstance(item, dict):

                    extract_fields(
                        item,
                        prefix=field_name,
                        fields=fields
                    )

    return fields


def get_collection_schema(
    database_name: str,
    collection_name: str,
    sample_size: int = 100
):
    """
    Infer the schema of one MongoDB collection
    by sampling documents.
    """
    collection = client[database_name][collection_name]

    fields = {}

    for doc in collection.find().limit(sample_size):

        extract_fields(
            doc,
            fields=fields
        )

    return {
        field: sorted(types)
        for field, types in fields.items()
    }


def get_database_schema(
    database_name: str,
    sample_size: int = 100
):
    """
    Infer schemas for all collections in a MongoDB database.
    """
    db = client[database_name]

    schema = {}

    for collection_name in db.list_collection_names():

        schema[collection_name] = get_collection_schema(
            database_name=database_name,
            collection_name=collection_name,
            sample_size=sample_size
        )

    return schema


def schema_to_text(schema: dict):
    """
    Convert inferred schema into compact LLM-friendly text.
    """
    lines = []

    for collection_name, fields in schema.items():

        lines.append(f"Collection: {collection_name}")
        lines.append("Fields:")

        for field_name, field_types in fields.items():

            types = " | ".join(field_types)

            lines.append(
                f"  {field_name}: {types}"
            )

        lines.append("")

    return "\n".join(lines)


def execute_pipeline(
    database_name: str,
    collection_name: str,
    pipeline: list[dict]
):
    """
    Execute a MongoDB aggregation pipeline and
    return results, execution error details,
    and the equivalent MongoDB shell command.
    """

    mongo_command = pipeline_to_mongo_command(
        collection_name=collection_name,
        pipeline=pipeline
    )

    try:
        collection = client[
            database_name
        ][collection_name]

        results = list(
            collection.aggregate(pipeline)
        )

        return {
            "success": True,
            "mongo_command": mongo_command,
            "results": results,
            "error_type": None,
            "error_message": None
        }

    except PyMongoError as e:
        return {
            "success": False,
            "mongo_command": mongo_command,
            "results": None,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }


def pipeline_to_mongo_command(
    collection_name: str,
    pipeline: list[dict]
) -> str:
    """
    Convert a PyMongo aggregation pipeline into a MongoDB shell-style command.

    Example:
        db.cars_data.aggregate([
            {"$match": {"Weight": {"$gte": 3000}}},
            {"$group": {"_id": "$Year"}}
        ])
    """
    pipeline_text = json.dumps(
        pipeline,
        indent=2,
        default=str
    )

    return f"db.{collection_name}.aggregate({pipeline_text})"


def main():
    database_name = "car_1"
    collection_name = "cars_data"

    # 1. Test database schema extraction
    schema = get_database_schema(
        database_name=database_name,
        sample_size=100
    )

    print("=== DATABASE SCHEMA ===")
    print(schema)

    # 2. Test schema conversion to LLM-friendly text
    schema_text = schema_to_text(schema)

    print("\n=== SCHEMA TEXT ===")
    print(schema_text)

    # 3. Test collection-specific schema extraction
    collection_schema = get_collection_schema(
        database_name=database_name,
        collection_name=collection_name,
        sample_size=100
    )

    print("\n=== COLLECTION SCHEMA ===")
    print(collection_schema)

    # 4. Test valid pipeline execution
    pipeline = [
        {"$match": {"Weight": {"$gte": 3000, "$lte": 4000}}},
        {"$group": {"_id": "$Year"}}
    ]

    result = execute_pipeline(
        database_name=database_name,
        collection_name=collection_name,
        pipeline=pipeline
    )

    print("\n=== VALID PIPELINE RESULT ===")
    print(result)

    # 5. Test invalid pipeline / error handling
    invalid_pipeline = [
        {"$invalidStage": {"foo": "bar"}}
    ]

    error_result = execute_pipeline(
        database_name=database_name,
        collection_name=collection_name,
        pipeline=invalid_pipeline
    )

    print("\n=== INVALID PIPELINE RESULT ===")
    print(error_result)


if __name__ == "__main__":
    main()
