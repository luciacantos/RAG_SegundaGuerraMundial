import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ======================================================
# 1. Cargar API key desde .env y crear cliente OpenAI
# ======================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "No se ha encontrado OPENAI_API_KEY en el .env. "
        "Revisa que el archivo .env existe y que la variable está bien escrita."
    )

client = OpenAI(api_key=api_key)

# ======================================================
# 2. Rutas de datos (carpeta /data/processed relativa al proyecto)
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "ww2_chunks_plus_qa.jsonl"
EMB_PATH = PROCESSED_DIR / "ww2_embeddings_plus_qa.npy"

# Caché en memoria para no recargar disco en cada pregunta
_chunks_cache = None
_emb_matrix_cache = None


def cargar_chunks_y_embeddings():
    """
    Carga los chunks (JSONL) y la matriz de embeddings (Numpy) desde disco.
    Solo debería llamarse una vez; el resto de veces usamos la caché.
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de chunks: {CHUNKS_PATH}")

    if not EMB_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de embeddings: {EMB_PATH}")

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    emb_matrix = np.load(EMB_PATH)

    print(f"✅ Chunks cargados: {len(chunks)} | Embeddings: {emb_matrix.shape}")
    return chunks, emb_matrix


def get_chunks_and_embeddings():
    """
    Devuelve (chunks, emb_matrix) usando caché global.
    Así, aunque Streamlit ejecute varias veces, el proceso del servidor
    solo los carga de disco la primera vez.
    """
    global _chunks_cache, _emb_matrix_cache

    if _chunks_cache is None or _emb_matrix_cache is None:
        _chunks_cache, _emb_matrix_cache = cargar_chunks_y_embeddings()

    return _chunks_cache, _emb_matrix_cache


# ======================================================
# 3. Funciones de embeddings y recuperación
# ======================================================
def embed_text(text: str) -> np.ndarray:
    """Crea el embedding de una pregunta (query)."""
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def cosine_sim(query_vec: np.ndarray, docs_matrix: np.ndarray) -> np.ndarray:
    """Calcula similitud coseno entre el embedding de la query y todos los docs."""
    num = docs_matrix @ query_vec
    den = (np.linalg.norm(docs_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    return num / den


def recuperar_chunks(query: str, chunks, emb_matrix, k: int = 5):
    """Devuelve los top-k chunks más similares a la query."""
    q_emb = embed_text(query)
    sims = cosine_sim(q_emb, emb_matrix)

    top_idx = np.argsort(sims)[::-1][:k]

    resultados = []
    for idx in top_idx:
        c = chunks[int(idx)]
        resultados.append(
            {
                "score": float(sims[idx]),
                "text": c["text"],
                "source": c.get("source"),
                "page": c.get("page"),
            }
        )
    return resultados


# ======================================================
# 4. Generación de respuesta con el modelo de chat
# ======================================================
def generar_respuesta(query: str, context_chunks):
    """
    Genera la respuesta final usando OpenAI con un prompt
    adaptado a la Segunda Guerra Mundial.
    """
    trozos_formateados = []
    for i, c in enumerate(context_chunks):
        trozos_formateados.append(
            f"[Fragmento {i+1} | Fuente: {c.get('source', 'desconocida')} "
            f"pág. {c.get('page', '?')}]\n{c['text']}"
        )

    contexto = "\n\n".join(trozos_formateados)

    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un historiador experto en la Segunda Guerra Mundial. "
                "Respondes siempre en español, de manera breve y directa, con un máximo de 6 líneas. "
                "Tu única fuente de información es el contexto proporcionado. No inventes datos."
            ),
        },
        {
            "role": "user",
            "content": (
                "A continuación tienes fragmentos relevantes procedentes de PDFs históricos "
                "sobre la Segunda Guerra Mundial.\n\n"
                f"{contexto}\n\n"
                "Usa SOLO esta información para responder. Si la información no aparece, di: "
                "\"No existe información suficiente en los documentos para responder.\"\n\n"
                "Instrucciones para la respuesta:\n"
                "- Máximo 6 líneas.\n"
                "- Sin subtítulos.\n"
                "- Respuesta clara y concisa.\n"
                "- Incluye citas al final en formato (Fuente: <PDF>, pág. X).\n\n"
                f"Pregunta: {query}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=mensajes,
        max_completion_tokens=300,
    )

    return resp.choices[0].message.content


# ======================================================
# 5. Función principal que usará la app Streamlit
# ======================================================
def responder_pregunta(query: str, k: int = 5) -> str:
    """
    Función principal del RAG.
    - query: pregunta del usuario.
    - k: número de fragmentos a recuperar.
    """
    chunks, emb_matrix = get_chunks_and_embeddings()
    contexto = recuperar_chunks(query, chunks, emb_matrix, k=k)
    respuesta = generar_respuesta(query, contexto)
    return respuesta


# ======================================================
# 6. Prueba rápida en modo script (opcional)
# ======================================================
if __name__ == "__main__":
    pregunta = "Explica por qué la batalla de Stalingrado fue un punto de inflexión."
    print("❓ Pregunta:", pregunta)
    print("\n📚 Respuesta generada:\n")
    print(responder_pregunta(pregunta, k=5))
