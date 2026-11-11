"""
Representaciones Bag of Words / TF-IDF y Word2Vec
-------------------------------------------------
Funciones del notebook 07 (análisis con Word2Vec), siguiendo el laboratorio
guía "Lab BoW - Word2Vec - BETO":


"""

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parent.parent
RUTA_RESENAS = RAIZ / "data" / "processed" / "resenas_limpias.csv"
DIR_RESULTADOS = RAIZ / "data" / "results"
DIR_MODELOS = RAIZ / "data" / "models"
RUTA_EMBEDDINGS_W2V = DIR_RESULTADOS / "embeddings_word2vec.npz"
RUTA_TFIDF = DIR_RESULTADOS / "tfidf.npz"

# Nombres en español de las categorías (los géneros vienen en inglés del Proyecto 1)
GENEROS_ES = {"Action": "Acción", "Animation": "Animación", "Comedy": "Comedia",
              "Drama": "Drama", "Horror": "Terror"}
ORIGENES = {"imdb_kaggle": "traducido", "sensacine": "nativo"}


# ---------------------------------------------------------------------------
# Carga del corpus

def cargar_corpus(ruta=RUTA_RESENAS):
    """
    Lee resenas_limpias.csv y agrega las columnas de análisis en español:
    genero_es, polaridad (= valoracion) y origen (traducido / nativo).
    """
    df = pd.read_csv(ruta)
    df["genero_es"] = df["genero"].map(GENEROS_ES).fillna(df["genero"])
    df["polaridad"] = df["valoracion"]
    df["origen"] = df["fuente"].map(ORIGENES).fillna("traducido") \
        if "fuente" in df else "traducido"
    return df


def stopwords_es():
    """Stopwords del español de spaCy (no requiere descargar el modelo)."""
    from spacy.lang.es.stop_words import STOP_WORDS
    return set(STOP_WORDS)


# ---------------------------------------------------------------------------
# Tokenización

_SEPARADOR_ORACIONES = re.compile(r"(?<=[.!?¡¿;:])\s+|\s*\n\s*")
_PALABRA = re.compile(r"[a-záéíóúüñ]+(?:-[a-záéíóúüñ]+)*", re.IGNORECASE)


def tokenizar(texto):
    """Texto -> lista de oraciones; cada oración es una lista de palabras en minúscula."""
    oraciones = []
    for oracion in _SEPARADOR_ORACIONES.split(str(texto)):
        palabras = [p.lower() for p in _PALABRA.findall(oracion)]
        if palabras:
            oraciones.append(palabras)
    return oraciones


def preparar_oraciones(textos, min_frases=20, umbral_frases=10.0):
    """
    Tokeniza todas las reseñas y une expresiones frecuentes de dos palabras en un
    solo token con gensim Phrases (por ejemplo "obra maestra" -> "obra_maestra",
    "banda sonora" -> "banda_sonora"), para poder estudiarlas como una unidad.

    Devuelve (oraciones de todo el corpus, tokens por reseña, oraciones por
    reseña, modelo de frases).
    """
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser, Phrases

    por_resena = [tokenizar(t) for t in textos]
    todas = [o for oraciones in por_resena for o in oraciones]
    # conectores del español que pueden ir dentro de una frase ("efectos de sonido")
    conectores = frozenset({"de", "del", "la", "el", "los", "las", "y", "a"}) | ENGLISH_CONNECTOR_WORDS
    frases = Phraser(Phrases(todas, min_count=min_frases, threshold=umbral_frases,
                             connector_words=conectores))
    por_resena = [[frases[o] for o in oraciones] for oraciones in por_resena]
    oraciones = [o for oraciones_r in por_resena for o in oraciones_r]
    tokens = [[p for o in oraciones_r for p in o] for oraciones_r in por_resena]
    return oraciones, tokens, por_resena, frases


def oraciones_de(oraciones_por_resena, mascara):
    """Oraciones de las reseñas seleccionadas (para entrenar un modelo con un subconjunto)."""
    return [o for oraciones, elegir in zip(oraciones_por_resena, mascara) if elegir for o in oraciones]


# ---------------------------------------------------------------------------
# Bag of Words / TF-IDF

def construir_tfidf(textos, max_features=20000, min_df=5, max_df=0.8):
    """
    Matriz TF-IDF (dispersa) de las reseñas, sin stopwords del español.
    Devuelve (vectorizador, matriz).
    """
    vectorizador = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df,
                                   stop_words=sorted(stopwords_es()),
                                   token_pattern=r"(?u)\b[^\W\d_]{2,}\b", lowercase=True)
    return vectorizador, vectorizador.fit_transform(textos)


def dispersion(matriz):
    """Porcentaje de ceros de una matriz dispersa."""
    return (1 - matriz.nnz / (matriz.shape[0] * matriz.shape[1])) * 100


def terminos_caracteristicos_tfidf(matriz, vocabulario, etiquetas, n=12):
    """
    Términos con mayor peso TF-IDF medio en cada categoría, comparado con el
    resto de las reseñas (diferencia de medias). Devuelve {categoría: [(término, peso)]}.
    """
    etiquetas = np.asarray(etiquetas)
    resultado = {}
    for categoria in pd.unique(etiquetas):
        dentro = np.asarray(matriz[etiquetas == categoria].mean(axis=0)).ravel()
        fuera = np.asarray(matriz[etiquetas != categoria].mean(axis=0)).ravel()
        diferencia = dentro - fuera
        mejores = diferencia.argsort()[::-1][:n]
        resultado[categoria] = [(vocabulario[i], round(float(diferencia[i]), 4)) for i in mejores]
    return resultado


# ---------------------------------------------------------------------------
# Word2Vec

def entrenar_word2vec(oraciones, sg, vector_size=150, window=5, min_count=5,
                      epochs=10, semilla=1, workers=1):
    """
    Entrena Word2Vec con gensim. sg=0 -> CBOW, sg=1 -> Skip-Gram.
    Con workers=1 el entrenamiento es reproducible con la misma semilla.
    Devuelve (modelo, segundos de entrenamiento).
    """
    from gensim.models import Word2Vec

    inicio = time.time()
    modelo = Word2Vec(sentences=oraciones, sg=sg, vector_size=vector_size, window=window,
                      min_count=min_count, epochs=epochs, seed=semilla, workers=workers)
    return modelo, time.time() - inicio


def vecinos(modelo, palabra, n=10):
    """Palabras más similares (most_similar); lista vacía si la palabra no está en el vocabulario."""
    palabra = palabra.lower().replace(" ", "_")
    if palabra not in modelo.wv:
        return []
    return [(p, round(float(s), 3)) for p, s in modelo.wv.most_similar(palabra, topn=n)]


def tabla_vecinos(modelos, palabras, n=8):
    """
    DataFrame con los vecinos de cada palabra en cada modelo, para compararlos.
    modelos: {nombre: modelo}.
    """
    filas = []
    for palabra in palabras:
        fila = {"palabra": palabra}
        for nombre, modelo in modelos.items():
            v = vecinos(modelo, palabra, n)
            fila[nombre] = ", ".join(p for p, _ in v) if v else "(fuera del vocabulario)"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("palabra")


def analogia(modelo, positivas, negativas, n=5):
    """
    Analogía vectorial: suma de 'positivas' menos 'negativas' (most_similar).
    Por ejemplo actor - hombre + mujer -> positivas=["actor", "mujer"], negativas=["hombre"].
    Devuelve (resultados, palabras fuera del vocabulario).
    """
    faltan = [p for p in positivas + negativas if p not in modelo.wv]
    if faltan:
        return [], faltan
    return [(p, round(float(s), 3))
            for p, s in modelo.wv.most_similar(positive=positivas, negative=negativas, topn=n)], []


def similitudes_grupos(modelo, grupos):
    """
    Similitud media dentro de cada campo semántico y entre campos distintos.
    grupos: {campo: [palabras]}. Devuelve un DataFrame campo x campo.
    """
    vocab = {g: [p for p in ps if p in modelo.wv] for g, ps in grupos.items()}
    nombres = list(vocab)
    tabla = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
    for a in nombres:
        for b in nombres:
            sims = [modelo.wv.similarity(x, y) for x in vocab[a] for y in vocab[b] if x != y]
            tabla.loc[a, b] = np.mean(sims) if sims else np.nan
    return tabla


# ---------------------------------------------------------------------------
# Embeddings de reseñas y categorías

def vector_resena(tokens, modelo, stopwords):
    """Promedio de los vectores de las palabras de la reseña (sin stopwords)."""
    vectores = [modelo.wv[t] for t in tokens if t in modelo.wv and t not in stopwords]
    if not vectores:
        return np.zeros(modelo.vector_size, dtype=np.float32)
    return np.mean(vectores, axis=0)


def matriz_resenas(tokens_por_resena, modelo, stopwords):
    """Matriz (reseñas x dimensiones) con el embedding promedio de cada reseña."""
    return np.vstack([vector_resena(t, modelo, stopwords) for t in tokens_por_resena])


def centroides(matriz, etiquetas):
    """Vector promedio de cada categoría. Devuelve un DataFrame categoría x dimensiones."""
    etiquetas = np.asarray(etiquetas)
    categorias = list(pd.unique(etiquetas))
    filas = [np.asarray(matriz[etiquetas == c].mean(axis=0)).ravel() for c in categorias]
    return pd.DataFrame(np.vstack(filas), index=categorias)


def similitud_centroides(matriz, etiquetas, centrar=False):
    """
    Similitud coseno entre los centroides de las categorías.

    Los promedios de muchas reseñas se parecen mucho entre sí (todas hablan de
    cine), así que las similitudes quedan cerca de 1. Con centrar=True se resta
    el vector medio de todo el corpus antes de comparar: queda solo lo que
    distingue a cada categoría y las diferencias se ven con claridad.
    """
    c = centroides(matriz, etiquetas)
    valores = c.values
    if centrar:
        valores = valores - np.asarray(matriz.mean(axis=0)).ravel()
    return pd.DataFrame(cosine_similarity(valores), index=c.index, columns=c.index)


def similitud_media_entre(matriz, etiquetas, a, b, muestra=1500, semilla=1):
    """
    Similitud coseno media entre reseñas de la categoría a y de la b (con una
    muestra de cada una). Con a == b es la similitud media dentro del grupo.
    """
    rng = np.random.default_rng(semilla)
    etiquetas = np.asarray(etiquetas)
    ia = np.flatnonzero(etiquetas == a)
    ib = np.flatnonzero(etiquetas == b)
    ia = rng.choice(ia, min(muestra, len(ia)), replace=False)
    ib = rng.choice(ib, min(muestra, len(ib)), replace=False)
    sims = cosine_similarity(matriz[ia], matriz[ib])
    if a == b:
        sims = sims[~np.eye(len(ia), dtype=bool)]
    return float(np.mean(sims))


def frecuencias_por_categoria(tokens_por_resena, etiquetas, stopwords):
    """Counter de palabras (sin stopwords) de cada categoría."""
    frec = {}
    for tokens, etiqueta in zip(tokens_por_resena, etiquetas):
        frec.setdefault(etiqueta, Counter()).update(t for t in tokens if t not in stopwords)
    return frec


def vocabulario_caracteristico(modelo, matriz, etiquetas, tokens_por_resena, stopwords,
                               n=12, candidatos=400, min_frec=15, min_ratio=1.5):
    """
    Palabras características de cada categoría según su posición en el espacio:
    las más cercanas a la dirección (centroide de la categoría - centroide del
    resto) que además son relativamente más frecuentes en la categoría que en el
    resto (ratio de frecuencia relativa >= min_ratio).

    Devuelve {categoría: DataFrame(palabra, similitud, ratio, frecuencia)}.
    """
    etiquetas = np.asarray(etiquetas)
    frec = frecuencias_por_categoria(tokens_por_resena, etiquetas, stopwords)
    resultado = {}
    for categoria in pd.unique(etiquetas):
        dentro = np.asarray(matriz[etiquetas == categoria].mean(axis=0)).ravel()
        fuera = np.asarray(matriz[etiquetas != categoria].mean(axis=0)).ravel()
        f_dentro = frec[categoria]
        f_fuera = Counter()
        for otra, c in frec.items():
            if otra != categoria:
                f_fuera.update(c)
        total_dentro, total_fuera = sum(f_dentro.values()), sum(f_fuera.values())

        filas = []
        for palabra, sim in modelo.wv.similar_by_vector(dentro - fuera, topn=candidatos):
            if palabra in stopwords or f_dentro[palabra] < min_frec:
                continue
            ratio = (f_dentro[palabra] / total_dentro) / ((f_fuera[palabra] + 1) / total_fuera)
            if ratio >= min_ratio:
                filas.append({"palabra": palabra, "similitud": round(float(sim), 3),
                              "ratio_frecuencia": round(ratio, 2), "frecuencia": f_dentro[palabra]})
            if len(filas) == n:
                break
        resultado[categoria] = pd.DataFrame(filas)
    return resultado


# ---------------------------------------------------------------------------
# Guardado (para MongoDB y para el notebook 09)

def guardar_embeddings(ids, matriz, ruta=None):
    """Guarda los embeddings de las reseñas en un .npz con los arreglos ids y vectores."""
    ruta = ruta or RUTA_EMBEDDINGS_W2V
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, ids=np.asarray(ids), vectores=np.asarray(matriz, dtype=np.float32))


def guardar_tfidf(ids, matriz, vocabulario, ruta=None):
    """Guarda la matriz TF-IDF dispersa con los ids de las reseñas y el vocabulario."""
    from scipy import sparse

    ruta = ruta or RUTA_TFIDF
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(ruta, matriz.tocsr())
    np.savez_compressed(Path(ruta).with_name(Path(ruta).stem + "_meta.npz"),
                        ids=np.asarray(ids), vocabulario=np.asarray(vocabulario))
