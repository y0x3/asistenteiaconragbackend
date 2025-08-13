import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"

HEADERS = {"Authorization": "Bearer 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"}

def generar_respuesta_hf(texto_entrada):
    payload = {"inputs": texto_entrada}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        print("Error response:", response.status_code, response.text)
        return f"Error al llamar a Hugging Face API: {response.status_code}"
