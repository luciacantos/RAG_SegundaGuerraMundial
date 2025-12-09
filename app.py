import streamlit as st

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="Chat Segunda Guerra Mundial",
    page_icon="🌍",
    layout="centered"
)

# =========================
# ESTILOS (CSS)
# =========================
CUSTOM_CSS = """
<style>
.main {
    background: radial-gradient(circle at top, #111827 0, #020617 55%);
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.chat-container {
    background: rgba(15, 23, 42, 0.9);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.chat-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 4px;
    background: linear-gradient(120deg, #22c55e, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    color: transparent;
}
.chat-subtitle {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-bottom: 12px;
}
.divider {
    border-bottom: 1px solid rgba(55, 65, 81, 0.7);
    margin: 8px 0 12px 0;
}
.stChatMessage {
    padding: 6px 4px;
}
.block-container {
    padding-top: 2rem !important;
}
textarea {
    border-radius: 999px !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# HISTORIAL DE MENSAJES
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Soy tu asistente sobre la Segunda Guerra Mundial. "
                       "Pregúntame lo que quieras 🌍📚"
        }
    ]


# ========= IMPORTAMOS TU RAG REAL =========
from src.rag_qa import responder_pregunta

def get_answer(user_input: str) -> str:
    """Conecta el chat de Streamlit con tu pipeline RAG."""
    return responder_pregunta(user_input, k=5)


# =========================
# INTERFAZ DE CHAT
# =========================
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Título y subtítulo
    st.markdown('<div class="chat-title">Chat · Segunda Guerra Mundial</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-subtitle">Haz una pregunta histórica. '
        'Ejemplo: <em>"¿Qué desencadenó el inicio de la Segunda Guerra Mundial?"</em></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta aquí...")

    if user_input:
        # Guardar y mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Obtener y mostrar respuesta
        answer = get_answer(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.markdown('</div>', unsafe_allow_html=True)
