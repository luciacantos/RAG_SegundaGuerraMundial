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
# 4. Generación de respuesta con PROMPT MEJORADO
# ======================================================
def expandir_siglas(texto: str) -> str:
    """
    Sustituye siglas por su explicación la primera vez que aparecen en el texto.
    """
    sustituciones = {
        "RAF": "RAF (Real Fuerza Aérea británica)",
        "Luftwaffe": "Luftwaffe (fuerza aérea alemana)",
        "Kriegsmarine": "Kriegsmarine (marina de guerra alemana)",
        "Wehrmacht": "Wehrmacht (fuerzas armadas alemanas)",
        "USAAF": "USAAF (Fuerza Aérea del Ejército de EE.UU.)",
    }

    for sigla, expansion in sustituciones.items():
        texto = texto.replace(sigla, expansion, 1)

    return texto


def generar_respuesta(query: str, context_chunks):
    """
    Genera la respuesta final usando OpenAI con el prompt mejorado
    (versión del cuaderno).
    """
    # --- Refuerzo semántico para preguntas ambiguas ---
    query_lower = query.lower()

    if "amante" in query_lower:
        query += " (No confundir con matrimonio o relación legal)"
    elif "mujer" in query_lower or "esposa" in query_lower:
        query += " (Especificar si hubo matrimonio legal y cuándo)"

    # --- Construir fragmentos de contexto ---
    partes = []
    for i, c in enumerate(context_chunks):
        partes.append(
            f"[Fragmento {i+1} | Fuente: {c.get('source', 'desconocida')} "
            f"pág. {c.get('page', '?')}]\n{c['text']}"
        )

    context_text = "\n\n".join(partes)

    mensajes = [
    {
        "role": "system",
        "content": (
            "Eres un asistente RAG para historia de la Segunda Guerra Mundial. "
            "Respondes SIEMPRE en español, claro y natural, con un máximo de 6 líneas. "
            "REGLA DE ORO: solo puedes usar la información del CONTEXTO. No uses conocimiento externo. "
            "Si la información no está en el contexto, di exactamente: "
            "\"No existe información suficiente en los documentos para responder.\" "
            "\n\n"
            "PRIORIDAD DE FUENTES:\n"
            "1) Si en el contexto aparece un fragmento cuya Fuente sea 'qa_manual', debes priorizarlo y basarte en él. "
            "No contradigas 'qa_manual'.\n"
            "2) Si no hay 'qa_manual' relevante, usa el resto de fragmentos.\n\n"
            "CONTROL DE PREMISAS:\n"
            "Si la pregunta contiene una premisa falsa o simplificadora (ej. 'causó directamente', 'fue la única causa'), "
            "NO la aceptes. Corrige la premisa primero (por ejemplo empezando con 'No,' o 'No exactamente,').\n\n"
            "CAUSALIDAD:\n"
            "Distingue siempre entre causa indirecta/factor contribuyente y causa directa/suficiente. "
            "Evita palabras como 'desencadenante' si implican causalidad directa, salvo que el contexto lo diga explícitamente.\n\n"
            "CITAS:\n"
            "Termina SIEMPRE con citas en el formato: (Fuente: <PDF/qa_manual>, pág. X). "
            "No inventes fuentes ni páginas; usa solo las que aparezcan en el contexto."
        ),
    },
    {
        "role": "user",
        "content": (
            f"{context_text}\n\n"
            "Instrucciones de salida:\n"
            "- Máximo 6 líneas.\n"
            "- Sin subtítulos.\n"
            "- Tono natural (no académico).\n"
            "- Si hay una respuesta de 'qa_manual' en el contexto, úsala como base.\n"
            "- Si falta evidencia suficiente, devuelve EXACTAMENTE: "
            "\"No existe información suficiente en los documentos para responder.\"\n"
            "- Añade citas al final.\n\n"
            f"Pregunta: {query}"
        ),
    },
]

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=mensajes,
        max_completion_tokens=300,
        temperature=0,
    )

    respuesta = resp.choices[0].message.content

    # --- Postproceso: expansión automática de siglas ---
    respuesta = expandir_siglas(respuesta)

    return respuesta


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
