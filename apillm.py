import os
from openai import OpenAI

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta OPENROUTER_API_KEY en variables de entorno...")

REFERER = os.environ.get("APP_URL", "http://localhost:5000")

MODEL = os.environ.get("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "Referer": REFERER,  # <-- sin HTTP-
        "X-Title": "prueba_tecnica_activamente_backend",
    },
)


def generar_respuesta(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        top_p=1.0,
    )
    return response.choices[0].message.content
