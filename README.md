<p align="center">
  <img src="docs/banner_rag.png" alt="RAG · Segunda Guerra Mundial — Asistente de preguntas y respuestas con recuperación semántica" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenAI-embeddings%20%7C%20LLM-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Vector%20store-NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
</p>

> Sistema **RAG (Retrieval-Augmented Generation)** que responde preguntas sobre la Segunda Guerra Mundial a partir de una base de conocimiento documental, combinando **búsqueda semántica** y un **modelo de lenguaje** para generar respuestas fundamentadas únicamente en las fuentes recuperadas.

---

## 🎯 Qué hace

El usuario hace una pregunta sobre la Segunda Guerra Mundial a través de una interfaz web y el sistema:

1. Convierte la pregunta en un *embedding* semántico.
2. Recupera los fragmentos más relevantes de una base de conocimiento histórica mediante similitud coseno.
3. Construye un *prompt* aumentado con esos fragmentos y genera la respuesta con un LLM, **basándose solo en el contexto recuperado** (no en el conocimiento "libre" del modelo), lo que reduce las alucinaciones y permite citar la fuente.

## 📚 Base de conocimiento

La base documental reúne **~40 documentos PDF** organizados en **7 categorías temáticas**, además de un conjunto de **preguntas y respuestas redactadas manualmente** que refuerzan las consultas más habituales:

| # | Categoría | Ejemplos de contenido |
|---|-----------|------------------------|
| 01 | Batallas | Stalingrado, Normandía, Midway, Kursk, Berlín, Iwo Jima… |
| 02 | Frentes | Frente oriental y occidental, Pearl Harbor, Operación Barbarroja, Afrika Korps… |
| 03 | Política y sociedad | Yalta, Potsdam, Múnich, Holocausto, Juicios de Núremberg, pactos… |
| 04 | Tecnología y armamento | Máquina Enigma, carros de combate, Kriegsmarine… |
| 05 | Operaciones aliadas | Desembarco de Normandía, Market-Garden, Torch, Bagratión, Dragoon… |
| 06 | Documentos generales | Visión global del conflicto |
| 07 | Biografías | Hitler, Stalin, Churchill, Roosevelt, Goebbels |

> 📝 **QA manual:** los archivos `qa_manual.jsonl` contienen preguntas y respuestas escritas a mano que se integran en la base vectorial junto a los fragmentos de los PDF, mejorando la calidad de la recuperación en las consultas frecuentes.

## 🧩 Arquitectura del RAG

El pipeline se divide en tres fases:

### 1 · Preparación del conocimiento *(offline)*
- **Fuentes:** ~40 PDFs sobre 7 ejes temáticos (batallas, frentes, política y sociedad, tecnología y armamento, operaciones aliadas, documentos generales y biografías).
- **Extracción y limpieza:** extracción de texto y metadatos, eliminación de ruido y unificación de formato.
- **Fragmentación:** división en bloques solapados (`chunk_size = 1000`, `chunk_overlap = 200`) para no perder contexto en los cortes.
- **Estructuración:** cada fragmento se guarda en JSONL con su texto y metadatos (fuente, página, tipo), junto al set de **preguntas-respuestas manuales (QA)**.
- **Vectorización:** cada fragmento se convierte en un vector con `text-embedding-3-large` (OpenAI) y se almacena en un **vector store propio** (matriz NumPy `float32`).

### 2 · Recuperación de información *(online)*
- La pregunta del usuario se transforma en *embedding* con el mismo modelo.
- Se compara con todos los vectores almacenados mediante **similitud coseno** (implementación propia en NumPy).
- Se seleccionan los **Top-K = 5** fragmentos más relevantes, cada uno con su texto, fuente, página y *score* de similitud.

### 3 · Generación de la respuesta
- Se construye un **prompt aumentado** combinando la pregunta original con los fragmentos recuperados.
- El LLM de OpenAI genera una respuesta coherente y fundamentada en ese contexto.
