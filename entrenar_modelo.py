import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# CONFIGURACIÓN

BASE_DIR = Path(__file__).resolve().parent

ARCHIVO_DATASET = BASE_DIR / "dataset_vocacional.csv"
ARCHIVO_MODELO = BASE_DIR / "modelo_vocacional.pkl"
ARCHIVO_COLUMNAS = BASE_DIR / "columnas_modelo.pkl"
ARCHIVO_PREGUNTAS = BASE_DIR / "preguntas_modelo.pkl"
ARCHIVO_IMPORTANCIAS = BASE_DIR / "importancia_variables.csv"
ARCHIVO_METRICAS = BASE_DIR / "metricas_modelo.txt"

TARGET = "codigo_area"

# Estas columnas NO son respuestas del cuestionario y no deben usarse
# como entradas del modelo. Agrega aquí cualquier otra columna administrativa.
COLUMNAS_EXCLUIDAS = {
    TARGET,
    "id_estudiante",
    "edad",
    "genero",
    "sexo",
    "curso",
    "ciudad",
    "areas_interes",
    "area_vocacional",
    "carrera_interes_actual",
    "carrera_normalizada",
    "carrera_elegida_hoy",
    "nombre",
    "apellido",
    "correo",
    "email",
    "fecha",
    "timestamp",
    "seguridad_carrera"

}

# Textos personalizados para las variables conocidas. Las variables nuevas
# recibirán automáticamente un texto creado a partir de su nombre.
PREGUNTAS_PERSONALIZADAS = {
    "gusto_matematicas": "Me gusta resolver problemas matemáticos.",
    "resolucion_problemas": "Tengo facilidad para resolver problemas complejos.",
    "gusto_numeros": "Disfruto trabajar con números y estadísticas.",
    "pensamiento_logico": "Considero que tengo un buen pensamiento lógico.",
    "analisis_soluciones": "Me gusta analizar información para encontrar soluciones.",
    "gusto_programacion": "Disfruto programar o aprender sobre tecnología.",
    "aprendizaje_tecnologia": "Tengo facilidad para aprender nuevas tecnologías.",
    "interes_funcionamiento": "Me interesa entender cómo funcionan las cosas.",
    "investigacion_cientifica": "Disfruto investigar temas científicos o tecnológicos.",
    "interes_ciencia": "Me interesa investigar temas científicos.",
    "gusto_experimentos": "Me gusta realizar experimentos.",
    "interes_salud": "Me interesa la salud y el bienestar de las personas.",
    "gusto_construccion": "Me gusta construir, reparar o manipular objetos.",
    "gusto_tecnico": "Disfruto realizar actividades prácticas y técnicas.",
    "gusto_diseno": "Me gusta diseñar o crear contenido visual.",
    "creatividad": "Considero que tengo una buena creatividad.",
    "expresion_creativa": "Me gusta expresar mis ideas de forma creativa.",
    "gusto_artistico": "Me gustan las actividades artísticas.",
    "gusto_escritura": "Disfruto escribir textos o historias.",
    "gusto_ensenar": "Disfruto enseñar o ayudar a otras personas.",
    "comunicacion": "Considero que tengo una buena capacidad de comunicación.",
    "gusto_oratoria": "Me gusta hablar en público.",
    "liderazgo": "Considero que tengo habilidades de liderazgo.",
    "trabajo_equipo": "Considero que trabajo bien en equipo.",
    "orientacion_personas": "Disfruto orientar, enseñar o trabajar con personas.",
    "interes_liderazgo": "Me interesa dirigir grupos o proyectos.",
    "liderazgo_grupos": "Me gusta liderar grupos o coordinar actividades.",
    "persuasion": "Disfruto proponer ideas y convencer a otras personas.",
    "organizacion_personal": "Considero que tengo una buena organización personal.",
    "orden_informacion": "Me gusta organizar información y mantener todo ordenado.",
    "interes_cuerpo_humano": "Me interesa aprender sobre el cuerpo humano.",
}


def crear_texto_pregunta(nombre_variable: str) -> str:
    """Crea una pregunta legible para una columna sin texto personalizado."""
    if nombre_variable in PREGUNTAS_PERSONALIZADAS:
        return PREGUNTAS_PERSONALIZADAS[nombre_variable]

    texto = nombre_variable.replace("_", " ").strip()
    return texto[:1].upper() + texto[1:] + "."

# CARGAR Y PREPARAR DATOS

def cargar_dataset() -> pd.DataFrame:
    if not ARCHIVO_DATASET.exists():
        raise FileNotFoundError(f"No se encontró el archivo:\n{ARCHIVO_DATASET}")

    df = pd.read_csv(ARCHIVO_DATASET, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    print("=" * 70)
    print("DATASET VOCACIONAL CARGADO")
    print("=" * 70)
    print(f"Registros originales: {len(df)}")
    print(f"Columnas originales: {len(df.columns)}")

    return df


def detectar_variables_cuestionario(df: pd.DataFrame) -> list[str]:
    """
    Usa automáticamente TODAS las columnas del cuestionario con respuestas
    numéricas en escala 1-5, excepto las columnas administrativas excluidas.
    """
    variables = []
    descartadas = []

    for columna in df.columns:
        if columna.lower() in COLUMNAS_EXCLUIDAS:
            continue

        serie_numerica = pd.to_numeric(df[columna], errors="coerce")
        valores_no_vacios = serie_numerica.dropna()

        if valores_no_vacios.empty:
            descartadas.append((columna, "sin valores numéricos"))
            continue

        proporcion_numerica = serie_numerica.notna().mean()
        proporcion_likert = valores_no_vacios.between(1, 5, inclusive="both").mean()

        # Se admite cierto margen por posibles datos vacíos o errores de digitación.
        if proporcion_numerica >= 0.70 and proporcion_likert >= 0.90:
            variables.append(columna)
        else:
            descartadas.append((columna, "no parece una respuesta Likert 1-5"))

    if not variables:
        raise ValueError(
            "No se detectaron variables del cuestionario en escala 1-5. "
            "Revisa el separador del CSV y los nombres de las columnas."
        )

    print(f"\nVariables del cuestionario detectadas: {len(variables)}")
    for numero, variable in enumerate(variables, start=1):
        print(f"  {numero:02d}. {variable}")

    if descartadas:
        print("\nColumnas no utilizadas como preguntas:")
        for columna, motivo in descartadas:
            print(f"  - {columna}: {motivo}")

    return variables


def preparar_datos(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, str]]:
    if TARGET not in df.columns:
        raise ValueError(f"El CSV no contiene la variable objetivo '{TARGET}'.")

    variables_modelo = detectar_variables_cuestionario(df)
    columnas_requeridas = variables_modelo + [TARGET]
    df = df[columnas_requeridas].copy()

    for columna in columnas_requeridas:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    for columna in variables_modelo:
        fuera_de_rango = (
            df[columna].notna()
            & ~df[columna].between(1, 5, inclusive="both")
        )
        cantidad = int(fuera_de_rango.sum())

        if cantidad:
            print(
                f"Advertencia: {cantidad} valor(es) fuera de 1-5 en "
                f"'{columna}'. Se reemplazarán con la mediana."
            )
            df.loc[fuera_de_rango, columna] = pd.NA

    X = df[variables_modelo].copy()
    y = df[TARGET].copy()
    preguntas = {
        variable: crear_texto_pregunta(variable)
        for variable in variables_modelo
    }

    distribucion = y.value_counts().sort_index()

    print(f"\nRegistros válidos: {len(df)}")
    print(f"Total de variables predictoras: {X.shape[1]}")
    print("\nDistribución de las áreas:")

    for codigo, cantidad in distribucion.items():
        porcentaje = cantidad / len(y) * 100
        print(f"Área {codigo}: {cantidad} registro(s) ({porcentaje:.2f}%)")

    if y.nunique() < 2:
        raise ValueError("El dataset necesita al menos dos áreas vocacionales.")

    if distribucion.min() < 2:
        raise ValueError(
            "Existe un área con menos de 2 registros. No es posible hacer "
            "una división estratificada."
        )

    return X, y, variables_modelo, preguntas

# MODELOS

def crear_modelos() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def crear_pipeline(modelo) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("modelo", modelo),
        ]
    )


def seleccionar_mejor_modelo(X: pd.DataFrame, y: pd.Series):
    cantidad_minima = int(y.value_counts().min())
    n_splits = min(3, cantidad_minima)

    if n_splits < 2:
        raise ValueError("No hay suficientes ejemplos para validación cruzada.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mejor_nombre = None
    mejor_modelo = None
    mejor_f1 = -1.0

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)

    for nombre, modelo in crear_modelos().items():
        pipeline = crear_pipeline(modelo)
        resultados = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_weighted": "f1_weighted",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=-1,
            error_score="raise",
        )

        accuracy_promedio = resultados["test_accuracy"].mean()
        f1_promedio = resultados["test_f1_weighted"].mean()
        balanceada_promedio = resultados["test_balanced_accuracy"].mean()

        print(f"\n{nombre}")
        print(f"  Accuracy promedio:   {accuracy_promedio:.2%}")
        print(f"  F1 ponderado:        {f1_promedio:.2%}")
        print(f"  Accuracy balanceada: {balanceada_promedio:.2%}")

        if f1_promedio > mejor_f1:
            mejor_f1 = f1_promedio
            mejor_nombre = nombre
            mejor_modelo = modelo

    print(f"\nMejor modelo: {mejor_nombre}")
    print(f"Mejor F1 ponderado: {mejor_f1:.2%}")

    return mejor_nombre, mejor_modelo

# ENTRENAR, EVALUAR Y GUARDAR

def entrenar_y_evaluar(
    X: pd.DataFrame,
    y: pd.Series,
    nombre_modelo: str,
    modelo,
    variables_modelo: list[str],
) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = crear_pipeline(modelo)
    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    accuracy_balanceada = balanced_accuracy_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones, average="weighted", zero_division=0)
    reporte = classification_report(y_test, predicciones, zero_division=0)
    matriz = confusion_matrix(y_test, predicciones, labels=sorted(y.unique()))

    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 70)
    print(f"Modelo: {nombre_modelo}")
    print(f"Variables utilizadas: {len(variables_modelo)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Accuracy balanceada: {accuracy_balanceada:.2%}")
    print(f"F1 ponderado: {f1:.2%}")
    print("\nReporte de clasificación:")
    print(reporte)
    print("Matriz de confusión:")
    print(matriz)

    contenido_metricas = (
        f"Modelo seleccionado: {nombre_modelo}\n"
        f"Registros totales: {len(X)}\n"
        f"Registros de entrenamiento: {len(X_train)}\n"
        f"Registros de prueba: {len(X_test)}\n"
        f"Variables utilizadas: {len(variables_modelo)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Accuracy balanceada: {accuracy_balanceada:.4f}\n"
        f"F1 ponderado: {f1:.4f}\n\n"
        f"Reporte de clasificación:\n{reporte}\n"
        f"Matriz de confusión:\n{matriz}\n"
    )
    ARCHIVO_METRICAS.write_text(contenido_metricas, encoding="utf-8")

    return pipeline


def guardar_importancias(
    pipeline: Pipeline,
    variables_modelo: list[str],
    preguntas: dict[str, str],
) -> None:
    modelo_entrenado = pipeline.named_steps["modelo"]

    if not hasattr(modelo_entrenado, "feature_importances_"):
        return

    importancias = pd.DataFrame(
        {
            "variable": variables_modelo,
            "importancia": modelo_entrenado.feature_importances_,
            "pregunta": [preguntas[var] for var in variables_modelo],
        }
    ).sort_values(by="importancia", ascending=False)

    importancias.to_csv(
        ARCHIVO_IMPORTANCIAS,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTOP 15 VARIABLES MÁS IMPORTANTES:")
    print(importancias[["variable", "importancia"]].head(15).to_string(index=False))


def guardar_archivos(
    pipeline: Pipeline,
    variables_modelo: list[str],
    preguntas: dict[str, str],
) -> None:
    joblib.dump(pipeline, ARCHIVO_MODELO)
    joblib.dump(variables_modelo, ARCHIVO_COLUMNAS)
    joblib.dump(preguntas, ARCHIVO_PREGUNTAS)

    print("\n" + "=" * 70)
    print("ARCHIVOS GENERADOS")
    print("=" * 70)
    print(f"Modelo: {ARCHIVO_MODELO.name}")
    print(f"Columnas: {ARCHIVO_COLUMNAS.name} ({len(variables_modelo)} variables)")
    print(f"Preguntas: {ARCHIVO_PREGUNTAS.name}")
    print(f"Importancias: {ARCHIVO_IMPORTANCIAS.name}")
    print(f"Métricas: {ARCHIVO_METRICAS.name}")


def main() -> None:
    try:
        df = cargar_dataset()
        X, y, variables_modelo, preguntas = preparar_datos(df)

        nombre_modelo, mejor_modelo = seleccionar_mejor_modelo(X, y)

        pipeline_evaluado = entrenar_y_evaluar(
            X,
            y,
            nombre_modelo,
            mejor_modelo,
            variables_modelo,
        )
        guardar_importancias(pipeline_evaluado, variables_modelo, preguntas)

        # Modelo final entrenado con todos los registros disponibles.
        pipeline_final = crear_pipeline(mejor_modelo)
        pipeline_final.fit(X, y)
        guardar_archivos(pipeline_final, variables_modelo, preguntas)

        print("\nEntrenamiento finalizado correctamente.")
        print(
            "La página web mostrará automáticamente todas las variables "
            "guardadas en columnas_modelo.pkl."
        )

    except Exception as error:
        print("\nERROR DURANTE EL ENTRENAMIENTO:")
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
