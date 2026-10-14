"""

Se usa pos_analysis.csv se usan las etiquetas de spaCy,
"""

import numpy as np
import pandas as pd

PUNTUACION = {"PUNCT", "SYM"}
CATEGORIAS_LEXICAS = {"NOUN", "VERB", "ADJ", "ADV"}

# Palabras de negación: las del enunciado (no, nunca, jamás, nada, ni) más
# los indefinidos negativos
NEGACIONES = {"no", "nunca", "jamás", "nada", "ni", "nadie", "ningún",
              "ninguno", "ninguna", "ningunos", "ningunas", "tampoco"}

# Nombres legibles de cada métrica, para tablas y gráficos.
NOMBRES_METRICAS = {
    "n_tokens": "Tokens (sin puntuación)",
    "densidad_lexica": "Densidad léxica (%)",
    "ratio_verbos_sustantivos": "Ratio verbos/sustantivos",
    "idx_verbos": "Verbos por 100 tokens",
    "idx_adjetivos": "Adjetivos por 100 tokens",
    "idx_negacion": "Negaciones por 100 tokens",
    "idx_pron_1": "Pronombres 1.ª persona por 100 tokens",
    "idx_pron_2": "Pronombres 2.ª persona por 100 tokens",
    "idx_pron_3": "Pronombres 3.ª persona por 100 tokens",
    "ttr": "Diversidad léxica (TTR, primeros 50 tokens)",
    "long_oracion": "Longitud media de oración (tokens)",
}


def cargar_tokens(ruta):
    """
    Lee pos_analysis.csv conservando los tokens tal como estan
    """
    return pd.read_csv(ruta, keep_default_na=False)


def sin_puntuacion(tokens, col_upos="upos_spacy"):
    """
    r"\w" = tokens que no son puntuación ni símbolos
    """
    es_palabra = tokens["token"].astype(str).str.contains(r"\w", regex=True)
    return tokens[~tokens[col_upos].isin(PUNTUACION) & es_palabra]


def ttr_bloque(lemas, tamano=50):
    """
    Diversidad léxica (type-token ratio) sobre los primeros lemas.

    El TTR baja siempre al crecer el texto (las palabras empiezan a
    repetirse)
    """
    lemas = list(lemas)
    if len(lemas) < tamano:
        return np.nan
    bloque = [lema.lower() for lema in lemas[:tamano]]
    return len(set(bloque)) / tamano


def metricas_por_resena(tokens, tamano_ttr=50):
    """
    Metricas morfosintácticas de cada reseña.

    Devuelve un DataFrame con una fila por id y las columnas de
    :data:NOMBRES_METRICAS:

    * densidad_lexica: (NOUN + VERB + ADJ + ADV) / tokens, en %.
    * ratio_verbos_sustantivos: VERB / NOUN (alto = estilo narrativo,
      bajo = descriptivo). NaN si la reseña no tiene sustantivos.
    * idx_verbos, idx_adjetivos: VERB y ADJ por cada 100 tokens.
    * idx_negacion: palabras de :NEGACIONES por cada 100 tokens.
    * idx_pron_1/2/3: pronombres (PRON) con Person=1/2/3 por cada
      100 tokens.
    * ttr: ver :func:ttr_bloque.
    * long_oracion: tokens por oración
    """
    t = sin_puntuacion(tokens).copy()
    t["morf"] = t["morf"].fillna("")
    upos = t["upos_spacy"]
    es_pron = upos == "PRON"

    t["lexica"] = upos.isin(CATEGORIAS_LEXICAS)
    t["verbo"] = upos == "VERB"
    t["sustantivo"] = upos == "NOUN"
    t["adjetivo"] = upos == "ADJ"
    t["negacion"] = t["token"].str.lower().isin(NEGACIONES)
    for persona in (1, 2, 3):
        t[f"pron_{persona}"] = es_pron & t["morf"].str.contains(f"Person={persona}")

    g = t.groupby("id")
    n = g.size()
    por_100 = lambda columna: g[columna].sum() / n * 100

    metricas = pd.DataFrame({
        "n_tokens": n,
        "densidad_lexica": por_100("lexica"),
        "ratio_verbos_sustantivos": g["verbo"].sum() / g["sustantivo"].sum().replace(0, np.nan),
        "idx_verbos": por_100("verbo"),
        "idx_adjetivos": por_100("adjetivo"),
        "idx_negacion": por_100("negacion"),
        "idx_pron_1": por_100("pron_1"),
        "idx_pron_2": por_100("pron_2"),
        "idx_pron_3": por_100("pron_3"),
        "ttr": g["lema"].apply(ttr_bloque, tamano=tamano_ttr),
        "long_oracion": n / g["oracion"].nunique(),
    })
    metricas.index.name = "id"
    return metricas.reset_index()


def resumen_por_grupo(df, col_grupo, metricas=None):
    """
    Promedio de cada métrica por grupo (género, valoración, periodo...), con
    el número de reseñas del grupo
    """
    metricas = metricas or [m for m in NOMBRES_METRICAS if m in df.columns]
    resumen = df.groupby(col_grupo, observed=True)[metricas].mean()
    resumen.insert(0, "n", df.groupby(col_grupo, observed=True).size())
    return resumen


def comparar_grupos(df, metrica, col_grupo, grupo_a, grupo_b):
    """
    Compara una métrica entre dos grupos de reseñas: media, mediana y número
    de reseñas de cada grupo, y la diferencia entre ambos (en unidades de la
    métrica y en % respecto al grupo B).
    """
    a = df.loc[df[col_grupo] == grupo_a, metrica].dropna()
    b = df.loc[df[col_grupo] == grupo_b, metrica].dropna()
    return {
        "métrica": NOMBRES_METRICAS.get(metrica, metrica),
        "grupo A": grupo_a, "media A": a.mean(), "mediana A": a.median(), "n A": len(a),
        "grupo B": grupo_b, "media B": b.mean(), "mediana B": b.median(), "n B": len(b),
        "diferencia (A − B)": a.mean() - b.mean(),
        "diferencia (%)": (a.mean() - b.mean()) / b.mean() * 100,
    }


def mostrar(resultados):
    """
    Convierte el resultado de comparar_grupos (uno o una lista) en una tabla.
    """
    return pd.DataFrame(resultados if isinstance(resultados, list) else [resultados])


# ---------------------------------------------------------------------------
# Comparación NLTK vs spaCy

PARES_CRITERIO = {("AUX", "VERB"), ("VERB", "AUX"),
                  ("SCONJ", "PRON"), ("PRON", "SCONJ"),
                  ("NUM", "DET")}


def concordancia_global(tokens):
    """ % de tokens en que NLTK y spaCy coinciden."""
    t = sin_puntuacion(tokens)
    return (t["upos_spacy"] == t["upos_nltk"]).mean() * 100


def concordancia_por_categoria(tokens):
    """

    * % coincide (según spaCy): de los tokens que spaCy etiqueta como X,
      qué porcentaje etiqueta NLTK también como X.
    * % coincide (según NLTK): lo mismo, partiendo de las etiquetas de
      NLTK.
    """
    t = sin_puntuacion(tokens)
    igual = t["upos_spacy"] == t["upos_nltk"]
    tabla = pd.DataFrame({
        "tokens spaCy": t["upos_spacy"].value_counts(),
        "% coincide (según spaCy)": igual.groupby(t["upos_spacy"]).mean() * 100,
        "tokens NLTK": t["upos_nltk"].value_counts(),
        "% coincide (según NLTK)": igual.groupby(t["upos_nltk"]).mean() * 100,
    })
    tabla[["tokens spaCy", "tokens NLTK"]] = tabla[["tokens spaCy", "tokens NLTK"]].fillna(0).astype(int)
    return tabla.sort_values("tokens spaCy", ascending=False)


def matriz_confusion(tokens, normalizar=True):
    """
    Matriz de confusión: filas = etiqueta de spaCy, columnas = etiqueta de
    NLTK
    """
    t = sin_puntuacion(tokens)
    return pd.crosstab(t["upos_spacy"], t["upos_nltk"],
                       normalize="index" if normalizar else False) * (100 if normalizar else 1)


def causa_probable(fila):
    """
    Pre-clasificación automática de una discrepancia según su causa probable.

    Es solo una primera aproximación para orientar la revisión manual:

    1. ``Palabra ausente del entrenamiento de NLTK``: la palabra no aparece
       en ``cess_esp``, así que NLTK solo pudo asignarle la etiqueta por
       defecto (NOUN) o deducirla del contexto.
    2. ``Diferencia de criterio entre esquemas``: el par de etiquetas está en
       :data:`PARES_CRITERIO`.
    3. ``Ambigüedad real``: el resto. Palabras conocidas cuya categoría
       depende del contexto, que una de las herramientas resolvió mal.

    Los **errores de traducción** no se pueden detectar automáticamente: se
    identifican en la revisión manual, comparando con el texto en inglés.
    """
    if not fila["en_cess"]:
        return "Palabra ausente del entrenamiento de NLTK"
    if (fila["upos_spacy"], fila["upos_nltk"]) in PARES_CRITERIO:
        return "Diferencia de criterio entre esquemas"
    return "Ambigüedad real"


def discrepancias(tokens):
    """
    Tokens en los que NLTK y spaCy no coinciden, con la
    oración completa como contexto y su causa probable.
    """
    oraciones = (tokens.groupby(["id", "oracion"])["token"]
                 .apply(" ".join).rename("contexto"))
    t = sin_puntuacion(tokens)
    d = t[t["upos_spacy"] != t["upos_nltk"]].copy()
    d["causa_probable"] = d.apply(causa_probable, axis=1)
    return d.join(oraciones, on=["id", "oracion"])
