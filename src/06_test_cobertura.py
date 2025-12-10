# 06_test_cobertura.py
# Test de cobertura de PDFs para el RAG de la Segunda Guerra Mundial

from rag_qa import responder_pregunta


PREGUNTAS = [
    # =========================
    # 1) Segunda_Guerra_Mundial.pdf
    # =========================
    # Básicas
    "¿Cuáles fueron las causas principales de la Segunda Guerra Mundial según los documentos?",
    "¿Por qué Alemania pudo expandirse rápidamente entre 1939 y 1941?",
    "¿Qué países conformaron las Potencias del Eje y los Aliados?",

    # Avanzadas
    "¿Cómo afectó la Gran Depresión al ascenso de los regímenes totalitarios?",
    "¿Qué papel tuvo la remilitarización de Renania en la expansión alemana?",
    "¿Qué errores cometieron Reino Unido y Francia antes de 1939, según los textos?",
    "¿Cómo afectó la Blitzkrieg a los primeros años de guerra?",

    # Trampa
    "¿Qué opinaba Churchill sobre la Operación Azul?",
    "¿Cuántos civiles murieron exactamente en toda la guerra según este documento?",
    "¿Qué día concreto firmó Hitler la orden final para iniciar la guerra?",

    # =========================
    # 2) Invasión_de_Polonia.pdf
    # =========================
    # Básicas
    "¿Por qué Alemania invadió Polonia en 1939?",
    "¿Cómo participó la Unión Soviética en la invasión de Polonia?",
    "¿Por qué la invasión de Polonia se considera el inicio de la Segunda Guerra Mundial?",

    # Avanzadas
    "¿Qué fallos tácticos cometió Polonia ante la invasión alemana?",
    "¿Qué papel tuvo la Luftwaffe en la campaña de Polonia?",
    "¿Qué acuerdos previos permitieron la partición de Polonia?",

    # Trampa
    "¿Qué ejército polaco defendió la ciudad de Stalingrado?",
    "¿Cuántos tanques Tiger I participaron en la invasión de Polonia?",
    "¿Qué operaciones polacas de contraataque aparecen documentadas en los textos?",

    # =========================
    # 3) Operación_Barbarroja.pdf
    # =========================
    # Básicas
    "¿Cuál era el objetivo principal de la Operación Barbarroja?",
    "¿Por qué los alemanes no lograron capturar Moscú?",
    "¿Qué errores logísticos cometió Alemania en el frente oriental?",

    # Avanzadas
    "¿Qué papel tuvo el ancho de vía ferroviaria en el fracaso alemán en Barbarroja?",
    "¿Qué decisiones personales de Hitler perjudicaron la ofensiva en la URSS?",
    "¿Qué importancia tuvo el invierno ruso de 1941–1942 para el desarrollo de la campaña?",
    "¿Qué parte del ejército alemán sufrió mayores bajas durante Barbarroja?",

    # Trampa
    "¿Cuándo se celebró el llamado 'Tratado de Minsk' según el documento?",
    "¿Qué ejército japonés ayudó a Alemania durante la Operación Barbarroja?",
    "¿Cuántos tanques Panther operaron en la ofensiva inicial de Barbarroja?",

    # =========================
    # 4) Batalla_de_Stalingrado.pdf
    # =========================
    # Básicas
    "¿Por qué Stalingrado tenía importancia estratégica para ambos bandos?",
    "¿Qué errores cometió Alemania en la batalla de Stalingrado?",
    "¿Por qué la batalla de Stalingrado se considera un punto de inflexión en la guerra?",

    # Avanzadas
    "¿Qué condiciones climáticas afectaron a los combates en Stalingrado?",
    "¿Qué papel desempeñó Vasili Záitsev en la batalla de Stalingrado?",
    "¿Qué plan soviético permitió rodear al 6.º Ejército alemán en Stalingrado?",
    "¿Cómo fue el suministro alemán por vía aérea durante el cerco de Stalingrado?",

    # Trampa
    "¿Qué día exacto se congeló por completo el Volga según el texto sobre Stalingrado?",
    "¿Cuántos aviones Focke-Wulf 190 participaron en la batalla de Stalingrado?",
    "¿Cuántos civiles sobrevivieron exactamente al bombardeo inicial del 23 de agosto de 1942?",

    # =========================
    # 5) Holocausto.pdf
    # =========================
    # Básicas
    "¿Qué ideas raciales sustentaban la ideología nazi según los documentos?",
    "¿Qué papel jugó Goebbels en la propaganda antisemita del régimen nazi?",
    "¿Cómo se justificó públicamente la persecución de los judíos en la Alemania nazi?",

    # Avanzadas
    "¿Qué exposiciones públicas usó el régimen para promover el antisemitismo?",
    "¿Qué relación establecía la propaganda nazi entre los judíos y el bolchevismo?",
    "¿Qué instituciones o publicaciones se dedicaban a producir propaganda racial y antisemita?",

    # Trampa
    "¿Cuántos campos de exterminio aparecen listados exactamente en el documento sobre el Holocausto?",
    "¿Qué porcentaje exacto de judíos alemanes apoyaba a Hitler según los textos?",
    "¿Qué fecha concreta aparece como inicio 'oficial' del Holocausto en el documento?",
]


def imprimir_respuesta(idx: int, pregunta: str, resultado):
    print("=" * 100)
    print(f"[{idx}] PREGUNTA:")
    print(pregunta)
    print("-" * 100)

    # Normalizamos el tipo de respuesta:
    if isinstance(resultado, dict):
        resp_text = (
            resultado.get("respuesta")
            or resultado.get("answer")
            or str(resultado)
        )
    else:
        resp_text = str(resultado)

    print("RESPUESTA:")
    print(resp_text)
    print()


def main():
    total = len(PREGUNTAS)
    print("🔍 Test de cobertura del RAG — Segunda Guerra Mundial")
    print(f"Total de preguntas: {total}\n")

    for i, pregunta in enumerate(PREGUNTAS, start=1):
        try:
            resultado = responder_pregunta(pregunta)
        except Exception as e:
            print("=" * 100)
            print(f"[{i}] PREGUNTA (ERROR AL LLAMAR AL MODELO):")
            print(pregunta)
            print("-" * 100)
            print("ERROR:")
            print(e)
            print()
            continue

        imprimir_respuesta(i, pregunta, resultado)


if __name__ == "__main__":
    main()
