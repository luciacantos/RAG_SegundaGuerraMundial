from pathlib import Path
from rag_qa import responder_pregunta


PREGUNTAS_TEST = [
    # ===== 1. Batalla de Inglaterra: siglas + foco =====
    "¿Por qué fue importante la batalla de Inglaterra para frenar la expansión alemana?",
    "¿Qué buscaba exactamente la Luftwaffe en la batalla de Inglaterra?",
    "¿Qué papel jugó la RAF en la batalla de Inglaterra?",

    # ===== 2. Frente oriental: Barbarroja / Moscú / Stalingrado =====
    "Explica de forma breve qué era la Operación Barbarroja y cuál era su objetivo principal.",
    "¿Por qué fracasó el intento alemán de capturar Moscú según los documentos?",
    "¿Por qué la batalla de Stalingrado se considera un punto de inflexión en la guerra?",
    "¿Qué errores cometió Alemania en la batalla de Stalingrado?",

    # ===== 3. África del Norte =====
    "¿Qué era el Afrika Korps y qué objetivo tenía en el norte de África?",
    "¿Por qué fue tan importante la segunda batalla de El Alamein?",

    # ===== 4. Pacífico (Pearl Harbor, Midway, bombas atómicas) =====
    "¿Qué ocurrió en el ataque a Pearl Harbor y qué consecuencias tuvo?",
    "¿Por qué la batalla de Midway fue un punto de inflexión en la guerra del Pacífico?",
    "¿Qué cuentan los documentos sobre los bombardeos atómicos de Hiroshima y Nagasaki?",

    # ===== 5. Holocausto / ideología / propaganda =====
    "Según el documento del Holocausto, ¿cómo se define la ideología antisemita nazi?",
    "¿Qué papel tuvo Goebbels en la propaganda antisemita del régimen nazi?",
    "¿Qué relación establecía la propaganda nazi entre los judíos y el bolchevismo?",

    # ===== 6. Diplomacia y final de la guerra =====
    "¿Qué decisiones clave se tomaron en la conferencia de Yalta?",
    "¿Qué acuerdos o tensiones importantes aparecen en la conferencia de Potsdam?",
    "¿Qué objetivos tenían los Juicios de Núremberg según el documento?",

    # ===== 7. Tecnología y armamento =====
    "¿Qué era la máquina Enigma y por qué fue tan importante descifrarla?",
    "¿Qué tipos de carros de combate alemanes se mencionan y qué rasgos destacan de ellos?",

    # ===== 8. Preguntas donde debería responder 'no hay información suficiente' =====
    "¿Cuántos tanques Tiger I participaron exactamente en la invasión de Polonia?",
    "¿Qué día concreto se congeló por completo el río Volga durante la batalla de Stalingrado?",
    "¿Qué porcentaje exacto de la población alemana apoyaba a Hitler según los documentos?",
]


def main():
    print("✅ Test del RAG con el prompt actual")
    print(f"Total de preguntas: {len(PREGUNTAS_TEST)}")
    print("=" * 80)

    for i, pregunta in enumerate(PREGUNTAS_TEST, start=1):
        print(f"\n[{i}] PREGUNTA:")
        print(pregunta)
        print("-" * 80)
        respuesta = responder_pregunta(pregunta)
        print("RESPUESTA:")
        print(respuesta)
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
