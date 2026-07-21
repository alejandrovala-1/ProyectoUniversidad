from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# CONFIGURACIÓN GENERAL

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "modelo_vocacional.pkl"
RUTA_COLUMNAS = BASE_DIR / "columnas_modelo.pkl"
RUTA_PREGUNTAS = BASE_DIR / "preguntas_modelo.pkl"

app = Flask(__name__)


# ÁREAS VOCACIONALES

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
        "Orientación profesional",
        "Talleres de habilidades",
    ],
    1: [
        "Ingeniería de Software",
        "Ciberseguridad",
        "Inteligencia Artificial",
        "Ciencia de Datos",
        "Desarrollo de Aplicaciones",
    ],
    2: [
        "Medicina",
        "Odontología",
        "Enfermería",
        "Fisioterapia",
        "Nutrición",
        "Farmacia",
        "Laboratorio Clínico",
    ],
    3: [
        "Ingeniería Civil",
        "Ingeniería Mecatrónica",
        "Ingeniería Industrial",
        "Ingeniería Electrónica",
        "Ingeniería Mecánica",
    ],
    4: [
        "Diseño Gráfico",
        "Animación Digital",
        "Diseño de Interiores",
        "Producción Audiovisual",
        "Artes Visuales",
    ],
    5: [
        "Psicología",
        "Derecho",
        "Educación",
        "Trabajo Social",
        "Comunicación Social",
        "Sociología",
    ],
    6: [
        "Administración de Empresas",
        "Economía",
        "Marketing",
        "Negocios Internacionales",
        "Contabilidad",
        "Finanzas",
    ],
    7: [
        "Biología",
        "Agronomía",
        "Veterinaria",
        "Ingeniería Ambiental",
        "Química",
        "Biotecnología",
    ],
}


EXPLICACIONES = {
    0: (
        "Tus respuestas no muestran todavía una preferencia "
        "suficientemente marcada por una sola área."
    ),
    1: (
        "Tu perfil presenta afinidad por la tecnología, "
        "el pensamiento lógico y la resolución de problemas."
    ),
    2: (
        "Tu perfil muestra interés por la salud, las ciencias "
        "y el bienestar de las personas."
    ),
    3: (
        "Tus respuestas reflejan interés por construir, diseñar "
        "y encontrar soluciones prácticas."
    ),
    4: (
        "Tu perfil destaca por la creatividad, la expresión "
        "artística y el diseño."
    ),
    5: (
        "Tus respuestas muestran interés por trabajar con personas, "
        "comunicar, enseñar y orientar."
    ),
    6: (
        "Tu perfil refleja habilidades de liderazgo, organización "
        "y gestión."
    ),
    7: (
        "Tus respuestas indican interés por la ciencia, "
        "la investigación y la naturaleza."
    ),
}


RECOMENDACIONES = {
    0: (
        "Explora cursos introductorios y conversa con profesionales "
        "de diferentes áreas."
    ),
    1: (
        "Prueba cursos de programación, robótica, desarrollo web "
        "o análisis de datos."
    ),
    2: (
        "Explora actividades de biología, primeros auxilios "
        "o voluntariado relacionado con el bienestar."
    ),
    3: (
        "Realiza proyectos de construcción, electrónica, "
        "dibujo técnico o robótica."
    ),
    4: (
        "Crea un portafolio con dibujos, diseños, fotografías "
        "o proyectos audiovisuales."
    ),
    5: (
        "Participa en debates, voluntariados, actividades educativas "
        "o proyectos comunitarios."
    ),
    6: (
        "Desarrolla pequeños proyectos de emprendimiento, liderazgo "
        "u organización."
    ),
    7: (
        "Participa en ferias científicas, proyectos ambientales "
        "o actividades de laboratorio."
    ),
}


ESCALA = {
    1: "Nada",
    2: "Poco",
    3: "Regular",
    4: "Mucho",
    5: "Totalmente",
}

# CARGA DE ARCHIVOS

def cargar_recursos():
    archivos_requeridos = [
        RUTA_MODELO,
        RUTA_COLUMNAS,
        RUTA_PREGUNTAS,
    ]

    faltantes = [
        archivo.name
        for archivo in archivos_requeridos
        if not archivo.exists()
    ]

    if faltantes:
        raise FileNotFoundError(
            "Faltan los siguientes archivos: "
            + ", ".join(faltantes)
        )

    modelo = joblib.load(RUTA_MODELO)
    columnas = joblib.load(RUTA_COLUMNAS)
    preguntas = joblib.load(RUTA_PREGUNTAS)

    if not isinstance(columnas, list) or not columnas:
        raise ValueError(
            "columnas_modelo.pkl debe contener una lista no vacía."
        )

    if not isinstance(preguntas, dict):
        raise ValueError(
            "preguntas_modelo.pkl debe contener un diccionario."
        )

    preguntas_faltantes = [
        columna
        for columna in columnas
        if columna not in preguntas
    ]

    if preguntas_faltantes:
        raise ValueError(
            "Faltan textos para estas preguntas: "
            + ", ".join(preguntas_faltantes)
        )

    return modelo, columnas, preguntas


MODELO = None
COLUMNAS = []
PREGUNTAS = {}
ERROR_CARGA = None

try:
    MODELO, COLUMNAS, PREGUNTAS = cargar_recursos()
except Exception as error:
    ERROR_CARGA = str(error)

# FUNCIONES AUXILIARES

def obtener_clases_modelo(modelo):
    if hasattr(modelo, "classes_"):
        return modelo.classes_

    if hasattr(modelo, "named_steps"):
        modelo_final = modelo.named_steps.get("modelo")

        if modelo_final is not None and hasattr(
            modelo_final,
            "classes_",
        ):
            return modelo_final.classes_

        for paso in reversed(
            list(modelo.named_steps.values())
        ):
            if hasattr(paso, "classes_"):
                return paso.classes_

    raise AttributeError(
        "No fue posible obtener las clases del modelo."
    )


def convertir_codigo(valor):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return valor


def obtener_nombre_area(codigo):
    codigo = convertir_codigo(codigo)

    return AREAS.get(
        codigo,
        f"Área desconocida ({codigo})",
    )


def clasificar_afinidad(probabilidad):
    porcentaje = probabilidad * 100

    if porcentaje >= 70:
        return "Afinidad muy marcada"

    if porcentaje >= 50:
        return "Afinidad marcada"

    if porcentaje >= 35:
        return "Afinidad moderada"

    return "Perfil vocacional diverso"

# VALIDAR RESPUESTAS

def validar_respuestas(formulario):
    datos = {}
    errores = []

    # Recorre todas las columnas del modelo.
    for columna in COLUMNAS:
        valor_texto = formulario.get(
            columna,
            "",
        ).strip()

        if not valor_texto:
            errores.append(
                f"Debes responder: {PREGUNTAS[columna]}"
            )
            continue

        try:
            valor = int(valor_texto)
        except ValueError:
            errores.append(
                f"La respuesta de "
                f"'{PREGUNTAS[columna]}' no es válida."
            )
            continue

        if valor not in ESCALA:
            errores.append(
                f"La respuesta de "
                f"'{PREGUNTAS[columna]}' debe estar entre 1 y 5."
            )
            continue

        datos[columna] = valor

    return datos, errores

# REALIZAR PREDICCIÓN

def realizar_prediccion(datos):
    estudiante = pd.DataFrame(
        [
            {
                columna: datos[columna]
                for columna in COLUMNAS
            }
        ],
        columns=COLUMNAS,
    )

    prediccion = MODELO.predict(estudiante)[0]
    codigo_predicho = convertir_codigo(prediccion)

    probabilidades = MODELO.predict_proba(
        estudiante
    )[0]

    clases = obtener_clases_modelo(MODELO)

    resultados = []

    for clase, probabilidad in zip(
        clases,
        probabilidades,
    ):
        codigo = convertir_codigo(clase)
        probabilidad = float(probabilidad)

        resultados.append(
            {
                "codigo": codigo,
                "area": obtener_nombre_area(codigo),
                "probabilidad": probabilidad,
                "porcentaje": round(
                    probabilidad * 100,
                    2,
                ),
            }
        )

    resultados.sort(
        key=lambda resultado: resultado["probabilidad"],
        reverse=True,
    )

    principal = resultados[0]

    segunda = (
        resultados[1]
        if len(resultados) > 1
        else None
    )

    diferencia = None
    perfil_combinado = False

    if segunda is not None:
        diferencia = round(
            principal["porcentaje"]
            - segunda["porcentaje"],
            2,
        )

        perfil_combinado = diferencia < 5

    return {
        "codigo_predicho": codigo_predicho,
        "principal": principal,
        "top_3": resultados[:3],
        "nivel_afinidad": clasificar_afinidad(
            principal["probabilidad"]
        ),
        "explicacion": EXPLICACIONES.get(
            principal["codigo"],
            "No hay una explicación disponible.",
        ),
        "carreras": CARRERAS.get(
            principal["codigo"],
            [
                "No hay carreras registradas "
                "para esta área."
            ],
        ),
        "recomendacion": RECOMENDACIONES.get(
            principal["codigo"],
            "Continúa explorando tus intereses.",
        ),
        "segunda": segunda,
        "diferencia": diferencia,
        "perfil_combinado": perfil_combinado,
        "total_variables": len(COLUMNAS),
    }

# RUTAS

@app.route("/", methods=["GET"])
def inicio():
    return render_template(
        "index.html",
        columnas=COLUMNAS,
        preguntas=PREGUNTAS,
        escala=ESCALA,
        total_preguntas=len(COLUMNAS),
        errores=[],
        respuestas={},
        error_carga=ERROR_CARGA,
    )


@app.route("/predecir", methods=["POST"])
def predecir():
    if ERROR_CARGA:
        return render_template(
            "index.html",
            columnas=COLUMNAS,
            preguntas=PREGUNTAS,
            escala=ESCALA,
            total_preguntas=len(COLUMNAS),
            errores=[],
            respuestas={},
            error_carga=ERROR_CARGA,
        ), 500

    datos, errores = validar_respuestas(
        request.form
    )

    if errores:
        return render_template(
            "index.html",
            columnas=COLUMNAS,
            preguntas=PREGUNTAS,
            escala=ESCALA,
            total_preguntas=len(COLUMNAS),
            errores=errores,
            respuestas=request.form,
            error_carga=None,
        ), 400

    resultado = realizar_prediccion(datos)

    return render_template(
        "resultado.html",
        resultado=resultado,
    )

# ERRORES

@app.errorhandler(404)
def pagina_no_encontrada(_error):
    return render_template(
        "error.html",
        titulo="Página no encontrada",
        mensaje="La dirección solicitada no existe.",
    ), 404


@app.errorhandler(500)
def error_interno(_error):
    return render_template(
        "error.html",
        titulo="Error interno",
        mensaje=(
            "Ocurrió un problema al procesar la solicitud. "
            "Revisa la consola del servidor."
        ),
    ), 500

# EJECUCIÓN
if __name__ == "__main__":
    app.run(debug=True)