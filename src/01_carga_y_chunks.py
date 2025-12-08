import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Rutas de entrada/salida
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def cargar_pdfs():
    """
    Carga todos los PDFs de data/raw como documentos de LangChain.
    """
    docs = []
    for pdf_path in RAW_DIR.glob("*.pdf"):
        print(f"📄 Cargando {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        # Añadir metadatos
        for d in pdf_docs:
            d.metadata["source"] = pdf_path.name
        docs.extend(pdf_docs)

    print(f"\n✅ Total de páginas cargadas: {len(docs)}")
    return docs


def crear_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """
    Divide los documentos en chunks usando RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)
    print(f"✅ Total de chunks generados: {len(chunks)}")

    # Mostrar un ejemplo
    ejemplo = chunks[0]
    print("\n--- Ejemplo de chunk ---")
    print(f"Source: {ejemplo.metadata.get('source')}")
    print(ejemplo.page_content[:400], "...")
    print("------------------------\n")

    return chunks


def guardar_chunks_jsonl(chunks, output_path):
    """
    Guarda los chunks en formato JSONL.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            item = {
                "id": i,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"💾 Archivo generado: {output_path}")


def main():
    docs = cargar_pdfs()
    chunks = crear_chunks(docs)
    output_jsonl = PROCESSED_DIR / "ww2_chunks.jsonl"
    guardar_chunks_jsonl(chunks, output_jsonl)


if __name__ == "__main__":
    main()
