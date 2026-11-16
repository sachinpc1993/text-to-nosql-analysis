import os

from dotenv import load_dotenv
from litellm import completion


load_dotenv()


def call_llm(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    **model_kwargs,
):
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **model_kwargs,
    }

    response = completion(**kwargs)

    return {
        "content": response.choices[0].message.content,
        "requested_model": model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "provider": getattr(response, "_hidden_params", {}).get(
            "custom_llm_provider"
        ),
        "usage": (
            response.usage.model_dump()
            if getattr(response, "usage", None)
            and hasattr(response.usage, "model_dump")
            else getattr(response, "usage", None)
        ),
    }