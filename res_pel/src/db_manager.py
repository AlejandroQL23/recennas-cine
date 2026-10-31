"""
Conexión con MongoDB
----------------------------------
Migra el corpus a la colección reseñas

La migración **reconstruye la colección completa** cada vez que se ejecuta:
borra lo que haya y vuelve a cargar todo desde los archivos del proyecto.
Los campos para los que todavía no hay datos se guardan como None (null) y
se llenan solos al volver a correr la migración cuando existan los archivos:

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
NOMBRE_BD = "resennas_cine"
NOMBRE_COLECCION = "resennas"

RUTA_RESENAS = RAIZ / "data" / "processed" / "resenas_limpias.csv"
RUTA_POS = RAIZ / "data" / "results" / "pos_analysis.csv"
RUTA_METRICAS = RAIZ / "data" / "results" / "metricas_por_resena.csv"
RUTA_W2V = RAIZ / "data" / "results" / "embeddings_word2vec.npz"
RUTA_BETO = RAIZ / "data" / "results" / "embeddings_beto.npz"
RUTA_SCRAPING = RAIZ / "data" / "processed" / "resenas_scraping.csv"


GENEROS_ES = {
    "Action": "Acción",
    "Animation": "Animación",
    "Comedy": "Comedia",
    "Drama": "Drama",
    "Horror": "Terror",
}

# Periodos del Proyecto 1 -> valores del esquema
PERIODOS = {
    "1998-2009": "antes_2010",
    "2010-2018": "2010_en_adelante",
}

# Decimales con que se guardan los embeddings
DECIMALES_EMBEDDINGS = 4


# ---------------------------------------------------------------------------
# Conexión

def conectar(uri=MONGO_URI, nombre_bd=NOMBRE_BD):
    """Devuelve la base de datos del proyecto (falla si MongoDB no responde)."""
    from pymongo import MongoClient

    cliente = MongoClient(uri, serverSelectionTimeoutMS=5000)
    cliente.admin.command("ping")
    return cliente[nombre_bd]


def obtener_coleccion(uri=MONGO_URI, nombre_bd=NOMBRE_BD,
                      nombre_coleccion=NOMBRE_COLECCION):
    """Colección de reseñas, lista para consultas."""
    return conectar(uri, nombre_bd)[nombre_coleccion]


# ---------------------------------------------------------------------------
# Utilidades

def valor(x):
    """Convierte NaN/NaT de pandas a None y los tipos de numpy a Python."""
    if x is None or (not isinstance(x, (list, dict, str)) and pd.isna(x)):
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    return x


def polaridad_de(calificacion):
    """Negativa (1-4), intermedia (5-6) o positiva (7-10), como en el Proyecto 1."""
    if calificacion is None or pd.isna(calificacion):
        return None
    if calificacion <= 4:
        return "negativa"
    if calificacion <= 6:
        return "intermedia"
    return "positiva"


def periodo_de(anio):
    """Periodo del esquema a partir del año de la reseña."""
    if anio is None or pd.isna(anio):
        return None
    return "antes_2010" if anio < 2010 else "2010_en_adelante"


# ---------------------------------------------------------------------------
# Lectura de los archivos opcionales

def cargar_pos(ruta=RUTA_POS):
    """
    Agrupa pos_analysis.csv por reseña.

    Devuelve un diccionario id -> {"nltk": [...], "spacy": [...]}, con un
    elemento por token en el orden del texto.
    """
    if not Path(ruta).exists():
        return {}
    tokens = pd.read_csv(ruta, keep_default_na=False)
    pos = {}
    for id_resena, grupo in tokens.groupby("id", sort=False):
        pos[int(id_resena)] = {
            "nltk": [{"oracion": int(o), "token": t, "upos": u}
                     for o, t, u in zip(grupo["oracion"], grupo["token"],
                                        grupo["upos_nltk"])],
            "spacy": [{"oracion": int(o), "token": t, "lema": l, "upos": u,
                       "tag": g, "morf": m}
                      for o, t, l, u, g, m in zip(grupo["oracion"], grupo["token"],
                                                  grupo["lema"], grupo["upos_spacy"],
                                                  grupo["tag_spacy"], grupo["morf"])],
        }
    return pos


def cargar_metricas(ruta=RUTA_METRICAS):
    """id -> métricas del esquema, tomadas de metricas_por_resena.csv."""
    if not Path(ruta).exists():
        return {}
    m = pd.read_csv(ruta)
    return {
        int(fila.id): {
            "densidad_lexica": valor(fila.densidad_lexica),
            "ratio_verbos_sustantivos": valor(fila.ratio_verbos_sustantivos),
            "indice_adjetivacion": valor(fila.idx_adjetivos),
            "indice_negacion": valor(fila.idx_negacion),
        }
        for fila in m.itertuples()
    }


def cargar_embeddings(ruta):
    """id -> vector redondeado, desde un .npz con los arreglos ids y vectores."""
    if not Path(ruta).exists():
        return {}
    datos = np.load(ruta)
    vectores = np.round(datos["vectores"].astype(float), DECIMALES_EMBEDDINGS)
    return {int(i): v.tolist() for i, v in zip(datos["ids"], vectores)}


# ---------------------------------------------------------------------------
# Construcción de documentos

def documento_base(id_resena, pos, w2v, beto):
    """Campos que se completan igual para cualquier origen de la reseña."""
    return {
        "pos_tags": pos.get(id_resena, {"nltk": None, "spacy": None}),
        "embeddings": {
            "word2vec_avg": w2v.get(id_resena),
            "beto_cls": beto.get(id_resena),
        },
    }


def documentos_proyecto1(resenas, pos, metricas, w2v, beto):
    """Documentos de las reseñas traducidas del Proyecto 1 (resenas_limpias.csv)."""
    fechas = pd.to_datetime(resenas["fecha"], format="%d %B %Y", errors="coerce")
    estrenos = pd.to_datetime(resenas["release_date"], errors="coerce")

    for fila, fecha, estreno in zip(resenas.itertuples(), fechas, estrenos):
        id_resena = int(fila.id)
        genero = GENEROS_ES.get(fila.genero, fila.genero)
        doc = {
            "id_resena": id_resena,
            "texto_es": fila.texto_es,
            "texto_en": fila.texto_en,
            "origen_texto": "traducido",
            "titulo_obra": None,          # el dataset de Kaggle no trae el título
            "tipo_obra": "pelicula",
            "generos": None,              # solo se conserva el género asignado
            "genero_principal": genero,
            "anio_estreno": valor(estreno.year) if pd.notna(estreno) else None,
            "calificacion": int(fila.calificacion),
            "polaridad": fila.valoracion,
            "periodo": PERIODOS.get(fila.periodo, fila.periodo),
            "fuente": "imdb_kaggle",
            "url_fuente": f"https://www.imdb.com/title/{fila.movie_id}/",
            "fecha_resena": valor(fecha),
            "fecha_recopilacion": None,
        }
        doc.update(documento_base(id_resena, pos, w2v, beto))
        doc["metricas"] = {"num_palabras": int(fila.n_palabras_es),
                           **metricas.get(id_resena, {
                               "densidad_lexica": None,
                               "ratio_verbos_sustantivos": None,
                               "indice_adjetivacion": None,
                               "indice_negacion": None})}
        yield doc



# ---------------------------------------------------------------------------
# Migración

def migrar(uri=MONGO_URI, nombre_bd=NOMBRE_BD, nombre_coleccion=NOMBRE_COLECCION,
           lote=500):
    """
    Borra la colección y la vuelve a cargar con todos los archivos disponibles.

    Devuelve un resumen con cuántas reseñas se cargaron y qué archivos se usaron.
    """
    resenas = pd.read_csv(RUTA_RESENAS)
    pos = cargar_pos()
    metricas = cargar_metricas()
    w2v = cargar_embeddings(RUTA_W2V)
    beto = cargar_embeddings(RUTA_BETO)

    documentos = list(documentos_proyecto1(resenas, pos, metricas, w2v, beto))

    bd = conectar(uri, nombre_bd)
    bd.drop_collection(nombre_coleccion)
    coleccion = bd[nombre_coleccion]
    for inicio in range(0, len(documentos), lote):
        coleccion.insert_many(documentos[inicio:inicio + lote], ordered=False)
    crear_indices(coleccion)

    return {
        "reseñas cargadas": coleccion.count_documents({}),
        "con pos_tags": len(pos),
        "con métricas": len(metricas),
        "con embeddings Word2Vec": len(w2v),
        "con embeddings BETO": len(beto),
        "nativas (scraping)": coleccion.count_documents({"origen_texto": "nativo"}),
    }


def crear_indices(coleccion):
    """Índices para los filtros y agrupaciones más usados en el análisis."""
    coleccion.create_index("id_resena", unique=True)
    for campo in ("genero_principal", "polaridad", "origen_texto", "periodo",
                  "fuente", "tipo_obra"):
        coleccion.create_index(campo)
