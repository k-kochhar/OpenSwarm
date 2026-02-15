from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VERIFY_STRUCTURE_PROMPT = (
    "You are a strict JSON/data structure validator. Given a piece of data, "
    "verify that it is well-formed and parseable. Respond with ONLY a JSON object: "
    '{"valid": true/false, "reason": "brief explanation"}. '
    "Do not include anything else in your response."
)


def chat(message: str, model: str = "gpt-4o") -> str:
    response = client.responses.create(
        model=model,
        input=message,
    )
    return response.output_text


def verify_structure(data: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=VERIFY_STRUCTURE_PROMPT,
        input=data,
    )
    return response.output_text
