"""
Embeddings contextuales con BETO

"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = RAIZ / "data" / "results"
RUTA_EMBEDDINGS_BETO = DIR_RESULTADOS / "embeddings_beto.npz"          # [CLS] (MongoDB: beto_cls)
RUTA_PARCIAL = DIR_RESULTADOS / "embeddings_beto_parcial.npz"

MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"


# ---------------------------------------------------------------------------
# Carga del modelo

def cargar_beto(modelo=MODELO_BETO, dispositivo=None):
    """
    Carga el tokenizador y el modelo BETO en modo evaluación.
    Usa GPU (cuda) si está disponible; si no, CPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(modelo)
    mt = AutoModel.from_pretrained(modelo).to(dispositivo)
    mt.eval()
    return {"tok": tok, "modelo": mt, "dispositivo": dispositivo}


def cargar_fill_mask(modelo=MODELO_BETO):
    """Pipeline de HuggingFace para predecir palabras enmascaradas ([MASK])."""
    from transformers import pipeline

    return pipeline("fill-mask", model=modelo)


def _ejecutar(beto, textos, max_tokens, **extra):
    """Tokeniza un lote y devuelve (entrada tokenizada, última capa oculta)."""
    import torch

    entrada = beto["tok"](textos, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_tokens, **extra)
    offsets = entrada.pop("offset_mapping", None)
    entrada = entrada.to(beto["dispositivo"])
    with torch.no_grad():
        salida = beto["modelo"](**entrada).last_hidden_state
    return entrada, salida.cpu().numpy(), offsets


# ---------------------------------------------------------------------------
# Polisemia(mucho significados): embedding de una palabra en contexto

def embedding_palabra(texto, palabra, beto, max_tokens=128):
    """
    Embedding contextual de 'palabra' dentro de 'texto': promedio de los vectores
    de la última capa de los sub-tokens que forman la primera aparición de la
    palabra. Devuelve None si la palabra no aparece en el texto.
    """
    m = re.search(rf"\b{re.escape(palabra)}\b", texto, re.IGNORECASE)
    if m is None:
        return None
    _, salida, offsets = _ejecutar(beto, [texto], max_tokens, return_offsets_mapping=True)
    indices = [i for i, (ini, fin) in enumerate(offsets[0].tolist())
               if fin > ini and ini < m.end() and fin > m.start()]
    if not indices:
        return None
    return salida[0, indices].mean(axis=0)


def embeddings_palabra(textos, palabra, beto):
    """Embeddings de 'palabra' en cada texto. Devuelve (matriz, índices de los textos válidos)."""
    vectores, validos = [], []
    for i, texto in enumerate(textos):
        v = embedding_palabra(texto, palabra, beto)
        if v is not None:
            vectores.append(v)
            validos.append(i)
    return np.vstack(vectores), validos


def contextos_del_corpus(df, palabra, n=60, ventana=25, semilla=1, col_texto="texto_es"):
    """
    Fragmentos reales del corpus que contienen 'palabra' (hasta n reseñas, una
    aparición por reseña)
    """
    patron = re.compile(rf"\b{re.escape(palabra)}\b", re.IGNORECASE)
    con_palabra = df[df[col_texto].str.contains(patron)]
    muestra = con_palabra.sample(min(n, len(con_palabra)), random_state=semilla)
    fragmentos = []
    for texto in muestra[col_texto]:
        palabras = texto.split()
        pos = next(i for i, p in enumerate(palabras) if patron.search(p))
        fragmentos.append(" ".join(palabras[max(0, pos - ventana):pos + ventana + 1]))
    return muestra.assign(fragmento=fragmentos)


def similitud_entre_sentidos(oraciones_por_sentido, palabra, beto):
    """
    Similitud coseno media entre usos de la palabra con el mismo sentido y con
    sentidos distintos
    """
    vectores = {s: embeddings_palabra(o, palabra, beto)[0] for s, o in oraciones_por_sentido.items()}
    sentidos = list(vectores)
    tabla = pd.DataFrame(index=sentidos, columns=sentidos, dtype=float)
    for a in sentidos:
        for b in sentidos:
            sims = cosine_similarity(vectores[a], vectores[b])
            if a == b:
                sims = sims[~np.eye(len(sims), dtype=bool)]
            tabla.loc[a, b] = float(np.mean(sims))
    return tabla, vectores


# ---------------------------------------------------------------------------
# Embeddings de reseñas completas

def embeddings_resenas(textos, beto, lote=16, max_tokens=512, ruta_parcial=None, cada=20,
                       progreso=True):
    """
    Embedding [CLS] de cada reseña (como la sección 3.3 del laboratorio). Las
    reseñas se ordenan por longitud para que cada lote tenga textos de tamaño
    parecido y se procese más rápido.
    """
    from tqdm.auto import tqdm

    textos = [str(t) for t in textos]
    n, dim = len(textos), beto["modelo"].config.hidden_size
    cls = np.zeros((n, dim), dtype=np.float32)
    hechos = np.zeros(n, dtype=bool)
    # huella de los textos y del límite de tokens: solo se reanuda si coinciden
    huella = hashlib.md5(("\x00".join(textos) + f"|{max_tokens}").encode("utf-8")).hexdigest()
    if ruta_parcial and Path(ruta_parcial).exists():
        datos = np.load(ruta_parcial)
        if str(datos["huella"]) == huella:
            cls, hechos = datos["cls"], datos["hechos"]
            print(f"Reanudando: {hechos.sum()} de {n} reseñas ya procesadas")
        else:
            print("El archivo parcial es de otros textos o de otro max_tokens: se empieza de cero")

    orden = [i for i in sorted(range(n), key=lambda i: len(textos[i])) if not hechos[i]]
    lotes = [orden[i:i + lote] for i in range(0, len(orden), lote)]
    for k, indices in enumerate(tqdm(lotes, unit="lote", disable=not progreso)):
        _, salida, _ = _ejecutar(beto, [textos[i] for i in indices], max_tokens)
        for j, i in enumerate(indices):
            cls[i] = salida[j, 0]
            hechos[i] = True
        if ruta_parcial and ((k + 1) % cada == 0 or k == len(lotes) - 1):
            Path(ruta_parcial).parent.mkdir(parents=True, exist_ok=True)
            np.savez(ruta_parcial, cls=cls, hechos=hechos, huella=huella)
    return cls


def embedding_texto(texto, beto, max_tokens=512):
    """Embedding [CLS] de un texto suelto (por ejemplo, una consulta)."""
    return embeddings_resenas([texto], beto, lote=1, max_tokens=max_tokens, progreso=False)[0]


def buscar(consulta_vector, matriz, n=5):
    """Índices y similitudes de las n filas de 'matriz' más parecidas al vector de la consulta."""
    sims = cosine_similarity(np.asarray(consulta_vector).reshape(1, -1), matriz).ravel()
    mejores = sims.argsort()[::-1][:n]
    return mejores, sims[mejores]


# ---------------------------------------------------------------------------
# Masked Language Model

def lexico_polaridad(textos, polaridades, min_frec=20):
    """
    Léxico de polaridad del propio corpus: para cada palabra, el logaritmo del
    cociente entre su frecuencia relativa en reseñas positivas y en negativas.
    > 0 = típica de reseñas positivas, < 0 = típica de negativas.
    """
    frec = {"positiva": Counter(), "negativa": Counter()}
    for texto, polaridad in zip(textos, polaridades):
        if polaridad in frec:
            frec[polaridad].update(re.findall(r"[a-záéíóúüñ]+", str(texto).lower()))
    total = {p: sum(c.values()) for p, c in frec.items()}
    lexico = {}
    for palabra in set(frec["positiva"]) | set(frec["negativa"]):
        pos, neg = frec["positiva"][palabra], frec["negativa"][palabra]
        if pos + neg >= min_frec:
            lexico[palabra] = float(np.log(((pos + 1) / total["positiva"]) / ((neg + 1) / total["negativa"])))
    return lexico


def etiquetar_polaridad(palabra, lexico, umbral=0.5):
    """'positiva', 'negativa' o 'neutra' según el léxico del corpus."""
    valor = lexico.get(palabra.lower().strip())
    if valor is None:
        return "sin datos"
    if valor >= umbral:
        return "positiva"
    if valor <= -umbral:
        return "negativa"
    return "neutra"


def predecir_mascara(fill_mask, oraciones, lexico, n=5):
    """
    Predicciones de BETO para cada oración con [MASK], con su probabilidad y la
    polaridad de la palabra según el léxico del corpus. Devuelve un DataFrame.
    """
    filas = []
    for oracion in oraciones:
        for r in fill_mask(oracion, top_k=n):
            palabra = r["token_str"].strip()
            filas.append({"oración": oracion, "predicción": palabra,
                          "probabilidad": round(float(r["score"]), 4),
                          "polaridad en el corpus": etiquetar_polaridad(palabra, lexico)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Guardado (MongoDB y notebook 09)

def guardar_embeddings(ids, matriz_cls):
    """
    Guarda embeddings_beto.npz (vector [CLS], el campo beto_cls del esquema de
    MongoDB) con los arreglos ids y vectores.
    """
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUTA_EMBEDDINGS_BETO, ids=np.asarray(ids),
                        vectores=np.asarray(matriz_cls, dtype=np.float32))
