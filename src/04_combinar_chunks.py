import json
from pathlib import Path

# -----------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNKS_ORIG = PROCESSED_DIR / "ww2_chunks.jsonl"
CHUNKS_OUT = PROCESSED_DIR / "ww2_chunks_plus_qa.jsonl"

print("📌 Base del proyecto:", BASE_DIR)
print("📁 Leyendo chunks originales de:", CHUNKS_ORIG)
print("📁 Buscando archivos QA en:", RAW_DIR)


# -----------------------------------------
# CARGAR CHUNKS ORIGINALES
# -----------------------------------------

all_rows = []

with open(CHUNKS_ORIG, "r", encoding="utf-8") as f:
    for line in f:
        all_rows.append(json.loads(line))

print(f"✔ Chunks originales cargados: {len(all_rows)}")


# -----------------------------------------
# CARGAR TODOS LOS QA_*.JSONL AUTOMÁTICAMENTE
# -----------------------------------------

qa_files = list(RAW_DIR.glob("qa_*.jsonl"))

if not qa_files:
    print("⚠ No se encontraron archivos QA en data/raw/.")
else:
    print("📝 Archivos QA encontrados:")
    for f in qa_files:
        print("   →", f.name)

    for file in qa_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                all_rows.append(json.loads(line))

print(f"✔ Total de filas después de añadir QA: {len(all_rows)}")


# -----------------------------------------
# REASIGNAR IDS Y GUARDAR
# -----------------------------------------

CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
    for new_id, row in enumerate(all_rows):
        row["id"] = new_id
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("✅ Archivo combinado generado en:", CHUNKS_OUT)
print("📊 Total de entradas finales:", len(all_rows))
print("\n👉 Ahora ejecuta 02_embeddings_y_vectorstore.py para actualizar los embeddings.")
