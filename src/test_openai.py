from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API KEY cargada:", "OK" if api_key else "NO ENCONTRADA")

client = OpenAI(api_key=api_key)

resp = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hola, esto es una prueba de embedding sobre la Segunda Guerra Mundial.",
)

print("Dimensión del embedding:", len(resp.data[0].embedding))
