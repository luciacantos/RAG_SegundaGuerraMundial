"""
Script de test automático del RAG de la Segunda Guerra Mundial.

Ejecutar con:
    (.venv) python src/05_test_rag.py
"""

from rag_qa import responder_pregunta  # Import corregido

# =========================
# LISTA DE PREGUNTAS DE TEST
# =========================

PREGUNTAS_TEST = [
    # 1) Históricas directas
    "¿Cuál fue el objetivo principal de la Operación Barbarroja y por qué fracasó?",
    "¿Qué papel tuvo el petróleo del Cáucaso en la estrategia alemana?",
    "¿Por qué la invasión de Polonia se considera el inicio de la Segunda Guerra Mundial?",
    "¿Qué factores permitieron la resistencia soviética durante la batalla de Stalingrado?",
    "¿Qué errores logísticos cometió Alemania en el frente oriental?",
    "¿Qué impacto tuvo el invierno ruso en las operaciones alemanas?",
    "¿Cómo afectó la Luftwaffe al desarrollo temprano del conflicto?",

    # 2) Interpretativas / opinión
    "¿Fue Hitler un buen estratega militar o un obstáculo para su propio ejército?",
    "¿Podría Alemania haber ganado la guerra si no hubiera atacado la URSS?",
    "¿Qué alternativa estratégica habría cambiado el resultado del Frente Oriental?",
    "¿Por qué Stalingrado se considera más simbólica que otras batallas?",
    "¿Cómo habría cambiado la guerra si Japón hubiera atacado a la URSS en 1941?",

    # 3) Trampas / riesgo de alucinación
    "¿Cuántos panzer Tiger I participaron en la invasión de Polonia?",
    "¿Qué orden secreta dio Hitler el 12 de octubre de 1940 para detener la Operación Barbarroja?",
    "¿Qué comandante soviético dirigió personalmente la defensa del puente de Kalach?",
    "¿Qué rol tuvo España en la batalla del Volga?",

    # 4) Combinan varios docs
    "¿Cómo se relaciona la invasión de Polonia con la estrategia alemana en Barbarroja?",
    "¿Qué errores cometió Alemania tanto en Polonia como en Stalingrado?",
    "¿Qué similitudes hubo entre los avances iniciales en Polonia y en Barbarroja?",

    # 5) Cronología
    "¿Qué ofensivas alemanas ocurrieron justo antes de Stalingrado?",
    "¿Cómo progresó Alemania desde la invasión de Polonia hasta la Operación Azul?",

    # 6) Liderazgo / biografías
    "¿Cómo influyó la personalidad de Hitler en la conducción de la guerra?",
    "¿Qué papel tuvo Zhúkov en las contraofensivas soviéticas?",
    "¿Cuál fue la contribución de von Paulus a la derrota alemana en Stalingrado?",
    "¿Cómo era la relación entre Hitler y Manstein?",

    # 7) Raras / creativas
    "¿Qué habría pasado si Alemania hubiera capturado intactos los pozos petrolíferos del Cáucaso?",
    "¿Qué influencia psicológica tuvo Stalingrado en la moral alemana?",
    "¿Qué pensaban los generales alemanes sobre la obsesión de Hitler con Stalingrado?",
    "¿Existió algún plan alemán para capturar Moscú después de la derrota en Stalingrado?",
    "¿Qué papel jugó la propaganda soviética en la resistencia durante la batalla de Stalingrado?",

    # 8) Imposibles / alucinación controlada
    "¿Qué opinaba Hitler sobre la táctica de guerra de ratas en Stalingrado?",
    "¿Qué temperatura exacta hacía en el Volga el 16 de noviembre de 1942?",
    "¿Cuántos civiles sobrevivieron exactamente al bombardeo alemán del 23 de agosto de 1942?"
]


# =========================
# FUNCIÓN PARA IMPRIMIR RESULTADOS
# =========================

def imprimir_respuesta(idx, pregunta, resultado):
    print("=" * 100)
    print(f"[{idx}] PREGUNTA:")
    print(pregunta)
    print("-" * 100)

    # Manejar caso donde resultado es un string
    if isinstance(resultado, str):
        resp = resultado
    else:
        resp = (
            resultado.get("respuesta")
            or resultado.get("answer")
            or str(resultado)
        )

    print("RESPUESTA:")
    print(resp)
    print()

    # Mostrar fuentes si existen
    if isinstance(resultado, dict) and "chunks_usados" in resultado:
        print("FUENTES:")
        for ch in resultado["chunks_usados"]:
            src = ch.get("source", "¿?")
            page = ch.get("page", "?")
            print(f"  - {src} (pág. {page})")
        print()


# =========================
# FUNCIÓN PRINCIPAL
# =========================

def main():
    print("🔍 Test automático del RAG — Segunda Guerra Mundial")
    print(f"Total preguntas: {len(PREGUNTAS_TEST)}\n")

    for i, pregunta in enumerate(PREGUNTAS_TEST, start=1):
        try:
            resultado = responder_pregunta(pregunta)
        except Exception as e:
            print("=" * 100)
            print(f"[{i}] ERROR con la pregunta:\n{pregunta}")
            print("⚠", repr(e))
            continue

        imprimir_respuesta(i, pregunta, resultado)


if __name__ == "__main__":
    main()
