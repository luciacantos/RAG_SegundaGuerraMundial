from rag_qa import responder_pregunta

# ======================================================
# TEST EXTREMO - MIX TOTAL (para defensa del trabajo)
# ======================================================

PREGUNTAS_TEST_EXTREMO = [
    # -------------------------
    # 0) Básico
    # -------------------------
    "¿Cuándo empezó y cuándo terminó la Segunda Guerra Mundial según los documentos?",
    "¿Qué países formaban los Aliados y el Eje según el contexto?",
    "¿Qué fue el Blitzkrieg y por qué fue eficaz al principio de la guerra?",

    # -------------------------
    # 1) Hitler y entorno
    # -------------------------
    "¿Quién fue Adolf Hitler y qué papel tuvo en el inicio de la guerra?",
    "¿Qué relación tuvo Eva Braun con Adolf Hitler?",
    "¿Tuvo Hitler hijos o descendientes directos según los documentos?",

    # -------------------------
    # 2) Frente occidental
    # -------------------------
    "¿Qué fue el Día D y qué objetivo tenía la operación aliada en Normandía?",
    "¿Qué fue la evacuación de Dunkerque y por qué fue relevante?",
    "¿Qué fue la batalla de las Ardenas?",

    # -------------------------
    # 3) Batalla de Inglaterra
    # -------------------------
    "¿Por qué fue importante la batalla de Inglaterra?",
    "¿Qué papel jugó la RAF en la batalla de Inglaterra?",
    "¿Qué buscaba la Luftwaffe durante la batalla de Inglaterra?",

    # -------------------------
    # 4) Frente oriental
    # -------------------------
    "¿Qué fue la Operación Barbarroja?",
    "¿Por qué la batalla de Stalingrado fue un punto de inflexión?",
    "¿Qué errores cometió Alemania en Stalingrado según los documentos?",

    # -------------------------
    # 5) Pacífico
    # -------------------------
    "¿Qué ocurrió en el ataque a Pearl Harbor y qué consecuencias tuvo?",
    "¿Por qué la batalla de Midway fue clave en el Pacífico?",
    "¿Qué consecuencias tuvieron los bombardeos atómicos de Hiroshima y Nagasaki?",

    # -------------------------
    # 6) Holocausto y propaganda
    # -------------------------
    "¿Cómo define el contexto la ideología antisemita nazi?",
    "¿Qué papel tuvo Goebbels en la propaganda del régimen nazi?",

    # -------------------------
    # 7) Posguerra
    # -------------------------
    "¿Qué decisiones clave se tomaron en la conferencia de Yalta?",
    "¿Qué objetivos tenían los Juicios de Núremberg?",

    # -------------------------
    # 8) Trampa (no hay info)
    # -------------------------
    "¿Qué porcentaje exacto de población alemana apoyaba a Hitler según los documentos?",
    "¿Qué día exacto se congeló completamente el río Volga durante Stalingrado?",
    "¿Cuántos tanques Tiger I participaron exactamente en la invasión de Polonia?",
]


def main():
    print("✅ TEST EXTREMO RAG (src)")
    print(f"Total de preguntas: {len(PREGUNTAS_TEST_EXTREMO)}")
    print("=" * 90)

    for i, pregunta in enumerate(PREGUNTAS_TEST_EXTREMO, start=1):
        print(f"\n[{i}] PREGUNTA:")
        print(pregunta)
        print("-" * 90)
        respuesta = responder_pregunta(pregunta, k=5)
        print("RESPUESTA:")
        print(respuesta)
        print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
