import re


def clean_llm_output(response: str) -> str:
    """
    Clean common formatting artifacts from LLM output.

    Handles:
    - Leading/trailing whitespace
    - ```json ... ```
    - ```python ... ```
    - ``` ... ```
    - Other fenced code blocks

    Does not perform any task-specific parsing or validation.
    """

    response = response.strip()

    # Remove opening code fence such as:
    # ```json
    # ```python
    # ```
    response = re.sub(
        r"^```[a-zA-Z0-9_-]*\s*",
        "",
        response,
    )

    # Remove closing code fence
    response = re.sub(
        r"\s*```$",
        "",
        response,
    )

    return response.strip()