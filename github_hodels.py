import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
#token = os.environ["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"]

token = os.environ.get("GITHUB_TOKEN")
print(f"Token cargado: {token} (tipo: {type(token)})")  # Para verificar

if not token or not isinstance(token, str):
    raise ValueError("La variable de entorno GITHUB_TOKEN no está definida o no es una cadena")

credential = AzureKeyCredential(token)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    model=model
)

print(response.choices[0].message.content)

