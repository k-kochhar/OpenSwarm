import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def chat(message: str, model: str = "claude-sonnet-4-5-20250929", image_b64: str = None) -> str:
    if image_b64:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": message},
        ]
    else:
        content = message
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text

if __name__ == "__main__":
    
    print(chat("what is your purpose?"))