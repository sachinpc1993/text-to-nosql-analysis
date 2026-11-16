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


def main():

    prompt = "In one sentence, explain what MongoDB aggregation is."

    models = {
        "Llama 3.1 8B": "ollama/llama3.1:8b",

        "Gemma 4": "ollama/gemma4:latest",
        "Gemma 4 12B": "ollama/gemma4:12b",
        "Gemma 3 12B IT": "bedrock/google.gemma-3-12b-it",

        "Claude Sonnet 5":
            "bedrock/au.anthropic.claude-sonnet-5",
        "Claude Sonnet 4.6":
            "bedrock/au.anthropic.claude-sonnet-4-6",
        "Claude Opus 5":
            "bedrock/au.anthropic.claude-opus-5",
        "Claude Haiku 4.5":
            "bedrock/au.anthropic.claude-haiku-4-5-20251001-v1:0",

        "DeepSeek V3.2":
            "bedrock/deepseek.v3.2",

        "GPT-OSS 20B":"bedrock/openai.gpt-oss-20b-1:0",
        
        "GPT-3.5": "openai/gpt-3.5-turbo",
        "GPT-4": "openai/gpt-4",
        "GPT-5.6": "openai/gpt-5.6",

        "Mistral 7B": "bedrock/mistral.mistral-7b-instruct-v0:2",
    }
    
    for name, model in models.items():

        print(f"\n{'=' * 60}")
        print(f"MODEL: {name}")
        print(f"{'=' * 60}")

        try:
            response = call_llm(
                model=model,
                prompt=prompt,
                temperature=0.0,
            )

            print(response)

        except Exception as e:
            print(f"ERROR: {type(e).__name__}")
            print(str(e))


if __name__ == "__main__":
    main()