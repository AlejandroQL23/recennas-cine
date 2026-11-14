"""
Comparación BoW vs. Word2Vec vs. BETO (sección 5.3 del enunciado)
------------------------------------------------------------------
Las tres representaciones se evalúan sobre las mismas reseñas y con la misma
partición de entrenamiento y prueba (random_state = número de equipo):

* Clasificación con Logistic Regression (accuracy).
* Clustering con K-Means (Silhouette Score).
* Proyección 2D con t-SNE.

Los resultados se guardan en data/results/ para el dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = RAIZ / "data" / "results"
RUTA_CLASIFICACION = DIR_RESULTADOS / "comparacion_clasificacion.csv"
RUTA_CLUSTERING = DIR_RESULTADOS / "comparacion_clustering.csv"
RUTA_TSNE = DIR_RESULTADOS / "comparacion_tsne.csv"

REPRESENTACIONES = ["BoW (TF-IDF)", "Word2Vec", "BETO [CLS]"]

# tarea -> (columna con la categoría real, categorías que se usan; None = todas)
TAREAS = {
    "polaridad": ("polaridad", ["positiva", "negativa"]),
    "género": ("genero_es", None),
    "origen": ("origen", ["traducido", "nativo"]),
}


def cargar_representaciones(ids):
    """
    Carga las tres representaciones guardadas por los notebooks 07 y 08,
    ordenadas según 'ids' (las mismas reseñas en el mismo orden).
    """
    def ordenar(ids_archivo):
        posicion = {int(i): k for k, i in enumerate(ids_archivo)}
        return [posicion[int(i)] for i in ids]

    tfidf = sparse.load_npz(DIR_RESULTADOS / "tfidf.npz").tocsr()
    meta = np.load(DIR_RESULTADOS / "tfidf_meta.npz")
    w2v = np.load(DIR_RESULTADOS / "embeddings_word2vec.npz")
    beto = np.load(DIR_RESULTADOS / "embeddings_beto.npz")
    return {
        "BoW (TF-IDF)": tfidf[ordenar(meta["ids"])],
        "Word2Vec": w2v["vectores"][ordenar(w2v["ids"])],
        "BETO [CLS]": beto["vectores"][ordenar(beto["ids"])],
    }


def preparar(X):
    """Escala las representaciones densas; TF-IDF ya viene normalizado."""
    return X if sparse.issparse(X) else StandardScaler().fit_transform(X)


def reducir(X, dimensiones, semilla):
    """Reduce la dimensión: TruncatedSVD para TF-IDF (disperso) y PCA para las densas."""
    if sparse.issparse(X):
        return TruncatedSVD(n_components=dimensiones, random_state=semilla).fit_transform(X)
    return PCA(n_components=dimensiones, random_state=semilla).fit_transform(preparar(X))


def datos_tarea(df, tarea):
    """Posiciones de las reseñas que participan en la tarea y su categoría real."""
    columna, categorias = TAREAS[tarea]
    mascara = df[columna].isin(categorias) if categorias else df[columna].notna()
    return np.flatnonzero(mascara.values), df.loc[mascara, columna].values


def comparar_clasificacion(representaciones, df, semilla):
    """
    Accuracy de una Logistic Regression por tarea y representación.
    La partición 80/20 es la misma para las tres representaciones.
    """
    filas = []
    for tarea in TAREAS:
        indices, y = datos_tarea(df, tarea)
        i_train, i_test, y_train, y_test = train_test_split(
            indices, y, test_size=0.2, random_state=semilla, stratify=y)
        for nombre, X in representaciones.items():
            X = preparar(X)
            modelo = LogisticRegression(max_iter=2000, random_state=semilla)
            modelo.fit(X[i_train], y_train)
            filas.append({"tarea": tarea, "representacion": nombre,
                          "accuracy": accuracy_score(y_test, modelo.predict(X[i_test]))})
    return pd.DataFrame(filas)


def comparar_clustering(representaciones, df, semilla):
    """
    K-Means con k = número de categorías reales (5 géneros, 2 polaridades,
    2 orígenes) y Silhouette Score de los clusters, para cada representación.
    Se reduce a 100 dimensiones antes del clustering y el Silhouette se calcula
    sobre una muestra de 5 000 reseñas (sample_size), porque con todas es muy lento.
    """
    reducidas = {n: reducir(X, 100, semilla) for n, X in representaciones.items()}
    filas = []
    for tarea in TAREAS:
        indices, y = datos_tarea(df, tarea)
        k = len(set(y))
        for nombre, X in reducidas.items():
            grupos = KMeans(n_clusters=k, random_state=semilla, n_init=10).fit_predict(X[indices])
            filas.append({"categoria": tarea, "representacion": nombre, "k": k,
                          "silhouette": silhouette_score(X[indices], grupos, sample_size=5000,
                                                         random_state=semilla)})
    return pd.DataFrame(filas)


def proyectar_tsne(representaciones, df, semilla):
    """
    Proyección 2D de todas las reseñas con t-SNE para cada representación
    (antes se reduce a 50 dimensiones). Una fila por reseña y representación.
    """
    partes = []
    for nombre, X in representaciones.items():
        coords = TSNE(n_components=2, random_state=semilla).fit_transform(reducir(X, 50, semilla))
        partes.append(pd.DataFrame({
            "representacion": nombre, "x": coords[:, 0], "y": coords[:, 1],
            "genero_es": df["genero_es"].values, "polaridad": df["polaridad"].values,
            "origen": df["origen"].values, "titulo_obra": df["titulo_obra"].values,
        }))
    return pd.concat(partes, ignore_index=True)


def guardar_resultados(clasificacion, clustering, tsne):
    """Guarda los resultados en data/results/ (los lee el dashboard)."""
    clasificacion.to_csv(RUTA_CLASIFICACION, index=False)
    clustering.to_csv(RUTA_CLUSTERING, index=False)
    tsne.to_csv(RUTA_TSNE, index=False)
