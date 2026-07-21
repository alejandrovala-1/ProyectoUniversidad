import sys
from pathlib import Path

import joblib
import pandas as pd

# CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent

ARCHIVO_MODELO = BASE_DIR / "modelo_vocacional.pkl"
ARCHIVO_COLUMNAS = BASE_DIR / "columnas_modelo.pkl"
ARCHIVO_PREGUNTAS = BASE_DIR / "preguntas_modelo.pkl"

AREAS = {
    0: "Sin definir",
    1: "Tecnología",
    2: "Salud",
    3: "Ingeniería",
    4: "Arte y Diseño",
    5: "Ciencias Sociales",
    6: "Negocios y Administración",
    7: "Ciencias Naturales",
}

CARRERAS = {
    0: [
        "Exploración vocacional",
        "Asesoría de orientación profesional",
    ],
    1: [
        "Ingeniería de Software",
        "Ciencias de la Computación",
        "Desarrollo de Sistemas",
        "Ciberseguridad",
        "Inteligencia Artificial",
        "Análisis de Datos",
    ],
    2: [
        "Medicina",
        "Enfermería",
        "Odontología",
        "Psicología",
        "Fisioterapia",
        "Nutrición",
        "Bioanálisis",
        "Farmacia",
    ],
    3: [
        "Ingeniería Civil",
        "Ingeniería Industrial",
        "Ingeniería Mecánica",
        "Ingeniería Eléctrica",
        "Ingeniería Electrónica",
        "Ingeniería Mecatrónica",
    ],
    4: [
        "Diseño Gráfico",
        "Arquitectura",
        "Diseño de Interiores",
        "Artes Visuales",
        "Comunicación Audiovisual",
        "Animación Digital",
    ],
    5: [
        "Derecho",
        "Educación",
        "Trabajo Social",
        "Sociología",
        "Comunicación Social",
        "Relaciones Internacionales",
    ],
    6: [
        "Administración de Empresas",
        "Contabilidad",
        "Mercadeo",
        "Economía",
        "Finanzas",
        "Negocios Internacionales",
    ],
    7: [
        "Biología",
        "Química",
        "Física",
        "Ciencias Ambientales",
        "Geología",
        "Biotecnología",
    ],
}

EXPLICACIONES = {
    0: "Tus respuestas no muestran una preferencia claramente definida por una sola área.",
    1: "Tu perfil presenta afinidad con la tecnología, la programación y la resolución lógica de problemas.",
    2: "Tu perfil presenta afinidad con la salud, el bienestar humano y el trabajo de cuidado.",
    3: "Tu perfil presenta afinidad con el diseño de soluciones técnicas, la construcción y el razonamiento aplicado.",
    4: "Tu perfil presenta afinidad con la creatividad, el diseño y la expresión artística.",
    5: "Tu perfil presenta afinidad con la comunicación, la enseñanza y el trabajo con personas.",
    6: "Tu perfil presenta afinidad con la organización, el liderazgo y la gestión de recursos.",
    7: "Tu perfil presenta afinidad con la investigación, la ciencia y el estudio de fenómenos naturales.",
}

# CARGA DE RECURSOS

def cargar_recursos():
    archivos = [
        ARCHIVO_MODELO,
        ARCHIVO_COLUMNAS,
        ARCHIVO_PREGUNTAS,
    ]

    faltantes = [archivo.name for archivo in archivos if not archivo.exists()]

    if faltantes:
        raise FileNotFoundError(
            "Faltan los siguientes archivos:\n- " + "\n- ".join(faltantes)
        )

    modelo = joblib.load(ARCHIVO_MODELO)
    columnas = joblib.load(ARCHIVO_COLUMNAS)
    preguntas = joblib.load(ARCHIVO_PREGUNTAS)

    if not isinstance(columnas, list) or not columnas:
        raise ValueError(
            "columnas_modelo.pkl debe contener una lista no vacía."
        )

    if not isinstance(preguntas, dict):
        raise ValueError(
            "preguntas_modelo.pkl debe contener un diccionario."
        )

    faltan_preguntas = [
        columna for columna in columnas
        if columna not in preguntas
    ]

    if faltan_preguntas:
        raise ValueError(
            "Faltan textos de preguntas para estas variables:\n- "
            + "\n- ".join(faltan_preguntas)
        )

    return modelo, columnas, preguntas

# CUESTIONARIO

def pedir_respuesta(numero: int, total: int, pregunta: str) -> int:
    while True:
        print(f"\nPregunta {numero} de {total}")
        print(pregunta)
        print("1 = Muy en desacuerdo")
        print("2 = En desacuerdo")
        print("3 = Neutral")
        print("4 = De acuerdo")
        print("5 = Muy de acuerdo")

        valor = input("Respuesta (1-5): ").strip()

        try:
            respuesta = int(valor)
        except ValueError:
            print("Entrada inválida. Escribe un número entre 1 y 5.")
            continue

        if 1 <= respuesta <= 5:
            return respuesta

        print("La respuesta debe estar entre 1 y 5.")


def responder_cuestionario(
    columnas: list[str],
    preguntas: dict[str, str],
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CUESTIONARIO VOCACIONAL")
    print("=" * 70)
    print(f"Total de preguntas: {len(columnas)}")
    print("Responde todas las preguntas usando una escala de 1 a 5.")

    respuestas = {}

    for numero, columna in enumerate(columnas, start=1):
        respuestas[columna] = pedir_respuesta(
            numero=numero,
            total=len(columnas),
            pregunta=preguntas[columna],
        )

    # El orden de las columnas debe ser exactamente el mismo usado
    # durante el entrenamiento.
    return pd.DataFrame([respuestas], columns=columnas)

# PREDICCIÓN Y RESULTADOS

def obtener_codigo(valor):
    """Convierte la clase predicha a entero cuando sea posible."""
    try:
        return int(valor)
    except (TypeError, ValueError):
        return valor


def mostrar_resultados(modelo, datos: pd.DataFrame) -> None:
    prediccion = modelo.predict(datos)[0]
    codigo_principal = obtener_codigo(prediccion)

    if not hasattr(modelo, "predict_proba"):
        print("\nÁrea recomendada:")
        print(AREAS.get(codigo_principal, str(codigo_principal)))
        return

    probabilidades = modelo.predict_proba(datos)[0]
    clases = modelo.classes_

    resultados = []

    for clase, probabilidad in zip(clases, probabilidades):
        codigo = obtener_codigo(clase)
        resultados.append(
            {
                "codigo": codigo,
                "area": AREAS.get(codigo, str(codigo)),
                "afinidad": float(probabilidad) * 100,
            }
        )

    resultados.sort(
        key=lambda elemento: elemento["afinidad"],
        reverse=True,
    )

    principal = resultados[0]
    top_3 = resultados[:3]

    print("\n" + "=" * 70)
    print("RESULTADO VOCACIONAL")
    print("=" * 70)
    print(f"Área recomendada: {principal['area']}")
    print(f"Afinidad estimada: {principal['afinidad']:.2f}%")

    explicacion = EXPLICACIONES.get(
        principal["codigo"],
        "El modelo identificó esta área como la más compatible.",
    )
    print(f"\nExplicación:\n{explicacion}")

    print("\nTOP 3 DE ÁREAS")
    for posicion, resultado in enumerate(top_3, start=1):
        print(
            f"{posicion}. {resultado['area']}: "
            f"{resultado['afinidad']:.2f}%"
        )

    carreras = CARRERAS.get(
        principal["codigo"],
        ["No hay carreras registradas para esta área."],
    )

    print("\nCARRERAS SUGERIDAS")
    for carrera in carreras:
        print(f"- {carrera}")

    print(
        "\nNota: este resultado representa una estimación de afinidad "
        "y no sustituye una evaluación profesional."
    )

# EJECUCIÓN PRINCIPAL

def main() -> None:
    try:
        modelo, columnas, preguntas = cargar_recursos()

        print("=" * 70)
        print("PRUEBA DEL MODELO VOCACIONAL")
        print("=" * 70)
        print(f"Variables cargadas: {len(columnas)}")

        datos = responder_cuestionario(
            columnas=columnas,
            preguntas=preguntas,
        )

        mostrar_resultados(
            modelo=modelo,
            datos=datos,
        )

    except KeyboardInterrupt:
        print("\n\nPrueba cancelada por el usuario.")
        sys.exit(0)

    except Exception as error:
        print("\nERROR AL PROBAR EL MODELO:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()