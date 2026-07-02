"""
Shared OpenAI client for all crews.
Reads OPENAI_API_KEY from environment (loaded via .env).
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def chat(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Simple wrapper for a single OpenAI chat completion.
    Returns the assistant message as a plain string.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # Low temperature for stable, reproducible research outputs
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
