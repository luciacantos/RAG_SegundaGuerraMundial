from dotenv import load_dotenv
import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

# Cargar variables del .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ⬇️ AHORA USAMOS EL ARCHIVO COMBINADO
CHUNKS_PATH = PROCESSED_DIR / "ww2_chunks_plus_qa.jsonl"
EMB_PATH = PROCESSED_DIR / "ww2_embeddings_plus_qa.npy"


def cargar_chunks():
    """Lee el JSONL de chunks y lo devuelve como lista de dicts."""
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"✅ Chunks cargados: {len(chunks)}")
    return chunks


def embed_batch(texts):
    """Crea embeddings para una lista de textos."""
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def main():
    chunks = cargar_chunks()

    batch_size = 64
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i : i + batch_size]]
        batch_embs = embed_batch(batch_texts)
        all_embeddings.extend(batch_embs)

        print(f"🔢 Procesados {i + len(batch_embs)} / {len(chunks)} chunks")

    emb_array = np.array(all_embeddings, dtype="float32")
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb_array)

    print(f"\n💾 Embeddings guardados en: {EMB_PATH}")
    print(f"   Shape: {emb_array.shape}  (n_chunks, dimensiones)")


if __name__ == "__main__":
    main()
