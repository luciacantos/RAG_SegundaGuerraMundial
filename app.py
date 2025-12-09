import streamlit as st
from src.rag_qa import responder_pregunta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="WORLD WAR II CHATBOT",
    page_icon="🌍",
    layout="centered"
)

# =========================
# CSS - TEMA OSCURO SIMPLE
# =========================
CUSTOM_CSS = """
<style>

html, body, .stApp {
    background-color: #0b0b0d !important;
    color: #e5e5e5 !important;
    font-family: 'Inter', sans-serif;
}

/* TITULO ÉPICO */
.main-title {
    text-align: center;
    font-size: 3.4rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.20em;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #f97316, #facc15);
    -webkit-background-clip: text;
    color: transparent;
}

/* Subtítulo */
.subtitle {
    text-align: center;
    font-size: 1rem;
    color: #d1d5db;
    margin-bottom: 2rem;
}

/* Línea */
.divider {
    border-bottom: 1px solid #2a2a2f;
    margin: 1.4rem 0;
}

/* Burbujas */
.stChatMessage div[data-testid="stMarkdown"] {
    padding: 12px 16px !important;
    border-radius: 14px !important;
}

/* Asistente */
.stChatMessage[data-testid="stChatMessage"] div[data-testid="stMarkdown"] {
    background-color: #1a1a1d !important;
}

/* Usuario */
.stChatMessage[data-testid="stChatMessage-user"] div[data-testid="stMarkdown"] {
    background-color: #1e3a8a !important;
    color: white !important;
}

/* Input inferior (chat_input) */
textarea {
    background-color: #121218 !important;
    border: 1px solid #34343a !important;
    color: #e5e5e5 !important;
    border-radius: 999px !important;
}

textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# HISTORIAL
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hola, soy tu asistente sobre la Segunda Guerra Mundial. Pregúntame lo que quieras. 🕊️"
    }]

# =========================
# FUNCIÓN RAG
# =========================
def get_answer(q):
    return responder_pregunta(q, k=5)

# =========================
# UI
# =========================
st.markdown('<div class="main-title">WORLD WAR II</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pregúntame sobre batallas, líderes, fechas clave o cualquier aspecto de la Segunda Guerra Mundial.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (el bueno, el simple, el que funciona bien)
user_input = st.chat_input("Escribe tu pregunta aquí...")

if user_input:
    # Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Respuesta RAG
    answer = get_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
