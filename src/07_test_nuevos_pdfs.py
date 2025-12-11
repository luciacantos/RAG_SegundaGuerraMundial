# src/07_test_nuevos_pdfs.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np

from rag_qa import responder_pregunta


# =========================
# RUTAS Y CARGA BÁSICA
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNKS_PATH = PROCESSED_DIR / "ww2_chunks_plus_qa.jsonl"
EMB_PATH = PROCESSED_DIR / "ww2_embeddings_plus_qa.npy"


def cargar_datos():
    """Carga los chunks y los embeddings solo para comprobar que cuadran."""
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    emb_matrix = np.load(EMB_PATH)
    print(f"✅ Chunks cargados: {len(chunks)} | Embeddings: {emb_matrix.shape}")
    return chunks, emb_matrix


# =========================
# PREGUNTAS DE PRUEBA
# =========================
PREGUNTAS = [
    # --- 01_Batallas ---
    "¿Por qué la batalla de Inglaterra fue importante para frenar la expansión alemana?",
    "¿Qué factores explican que la batalla de Kursk sea tan relevante en la guerra acorazada?",
    "¿Por qué la batalla de Midway se considera un punto de inflexión en el Pacífico?",
    "¿Qué objetivos tenía la ofensiva alemana en la batalla de Moscú y por qué fracasó?",
    "¿Qué importancia tuvo la segunda batalla de El Alamein para el frente del norte de África?",
    "¿Cómo afectó el sitio de Leningrado a la población civil según el documento?",

    # --- 02_Frentes ---
    "¿Qué fue el Afrika Korps y qué papel jugó en la campaña del norte de África?",
    "¿Qué ocurrió en el ataque a Pearl Harbor y qué consecuencias tuvo para la guerra?",
    "¿Por qué fue tan importante la campaña en África del Norte para ambos bandos?",
    "¿Cómo se describe en los documentos la invasión de Polonia de 1939?",
    "¿Cuál era el objetivo principal de la Operación Barbarroja según los textos?",

    # --- 03_Política y sociedad ---
    "¿Qué se cuenta en el documento sobre los bombardeos atómicos de Hiroshima y Nagasaki?",
    "¿Qué decisiones principales se tomaron en la conferencia de Yalta?",
    "¿Qué acuerdos o tensiones aparecen en la conferencia de Potsdam?",
    "¿Cómo se describe en el documento del Holocausto la ideología antisemita nazi?",
    "¿Qué objetivos tenían los Juicios de Núremberg y qué tipo de delitos se juzgaron?",

    # --- 04_Tecnología y armamento ---
    "¿Qué tipos de carros de combate alemanes se mencionan y qué rasgos se destacan de ellos?",
    "¿En qué consistía la máquina Enigma y por qué fue tan importante descifrarla?",

    # --- 05_Operaciones aliadas ---
    "¿Cómo se desarrolló el desembarco de Normandía según el documento específico?",
    "¿Cuáles eran los objetivos de la Operación Market-Garden y por qué terminó fracasando?",

    # --- 06_Documentos generales ---
    "Según el documento 'Segunda Guerra Mundial', ¿cómo se resume la evolución global del conflicto?",
]


# =========================
# IMPRESIÓN BONITA
# =========================
def imprimir_respuesta(idx: int, pregunta: str, resultado):
    """
    Imprime la respuesta en un formato legible.
    Soporta dos formatos:
    - dict con claves 'respuesta' / 'answer' / 'sources'
    - string plano
    """
    print("=" * 100)
    print(f"[{idx}] PREGUNTA:")
    print(pregunta)
    print("-" * 100)

    if isinstance(resultado, dict):
        resp = (
            resultado.get("respuesta")
            or resultado.get("answer")
            or resultado
        )
    else:
        resp = resultado

    print("RESPUESTA:")
    print(resp)

    if isinstance(resultado, dict) and "sources" in resultado:
        fuentes = resultado["sources"]
        if isinstance(fuentes, list) and fuentes:
            print("\n📚 Fuentes que ha usado el modelo:")
            for s in fuentes:
                if isinstance(s, dict):
                    src = s.get("source", "¿?")
                    page = s.get("page")
                    if page is not None:
                        print(f"   - {src} (pág. {page})")
                    else:
                        print(f"   - {src}")
                else:
                    print(f"   - {s}")
    print()


# =========================
# MAIN
# =========================
def main():
    print("🔍 Test de NUEVOS PDFs del RAG — Segunda Guerra Mundial")
    print(f"Total de preguntas: {len(PREGUNTAS)}\n")

    # Solo para comprobar que todo está alineado
    cargar_datos()

    for i, pregunta in enumerate(PREGUNTAS, start=1):
        resultado = responder_pregunta(pregunta, k=5)
        imprimir_respuesta(i, pregunta, resultado)


if __name__ == "__main__":
    main()
