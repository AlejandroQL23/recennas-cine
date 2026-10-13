"""
Etapa 1 · Traducción automática inglés -> español
--------------------------------------------------
Se usa el modelo MarianMT Helsinki-NLP/opus-mt-en-es

* La traducción del corpus se guarda en un
  archivo parcial (20).

Etapa 2 · Limpieza
------------------
Normalización de los textos (HTML, espacios, símbolos repetidos por el
traductor), eliminación de reseñas vacías, duplicadas o mal traducidas y
creación de la columna valoracion.
"""

import html
import re
import time
from pathlib import Path

import pandas as pd

MODELO_TRADUCCION = "Helsinki-NLP/opus-mt-en-es"


# ---------------------------------------------------------------------------
# Preparación del texto en inglés

def asegurar_recursos_nltk(recursos=("tokenizers/punkt",
                                      "tokenizers/punkt_tab")):
    """
    Verificar descarga NLTK, sino las descarga. esto para que la traduccion sea mas sencilla y rapida.
    """
    import nltk

    for recurso in recursos:
        try:
            nltk.data.find(recurso)
        except LookupError:
            nltk.download(recurso.split("/")[-1], quiet=True)


def preparar_texto_en(texto):
    """

      Proceso: valida que entrada sea String
      -Elimina etiquetas HTML
      -Inserta un espacio cuando una oración termina pegada a la siguiente
      "impression.This film" -> "impression. This film"
      -Quita dobles espacios

    """
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"([.!?])([A-Z])", r"\1 \2", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def dividir_oraciones(texto, max_palabras=100):
    """
    Divide una reseña en oraciones aptas para el traductor.

    Si max_pal>100, corta en trozos
    para que el modelo no la trunque y se pierda texto.
    """
    from nltk.tokenize import sent_tokenize

    oraciones = []
    for oracion in sent_tokenize(preparar_texto_en(texto), language="english"):
        palabras = oracion.split()
        if len(palabras) <= max_palabras:
            oraciones.append(oracion)
        else:
            for i in range(0, len(palabras), max_palabras):
                oraciones.append(" ".join(palabras[i:i + max_palabras]))
    return oraciones


# ---------------------------------------------------------------------------
# Modelo de traducción

def cargar_traductor(modelo=MODELO_TRADUCCION, dispositivo=None):
    """
    si hay GPU usa cuda = + rapido, sino usa CPU
    tok = de texto a numeros
    carga el modelo
    """
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(modelo)
    mt = MarianMTModel.from_pretrained(modelo).to(dispositivo)
    mt.eval()
    return {"tok": tok, "modelo": mt, "dispositivo": dispositivo}


def traducir_oraciones(oraciones, traductor, lote=16, max_length=256,
                       num_beams=None):
    """
    Traduce una lista de oraciones y las devuelve en el mismo orden.

    Parámetros
    ----------
    oraciones : list[str]
        Oraciones en inglés.
    traductor : dict
        Resultado de func:cargar_traductor.
    lote : int
        Oraciones por lote. En CPU, 8-16 suele ser lo más eficiente.
    max_length : int
        Máximo de tokens de entrada; lo que sobrepase se trunca.
    num_beams : int | None
        Usa el valor del modelo (4), que da mejor calidad;

    PyTorch= pt
    return_tensors= indeica en que formato devolver los datos
    """
    import torch

    tok, mt, dispositivo = (traductor["tok"], traductor["modelo"],
                            traductor["dispositivo"])

    # Se ordenan por longitud para que cada lote tenga oraciones de tamaño
    # parecido y luego se restaura el orden.
    orden = sorted(range(len(oraciones)), key=lambda i: len(oraciones[i]))
    traducidas = [""] * len(oraciones)

    opciones = {} if num_beams is None else {"num_beams": num_beams}
    for inicio in range(0, len(orden), lote):
        indices = orden[inicio:inicio + lote]
        entrada = tok([oraciones[i] for i in indices], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_length).to(dispositivo)
        with torch.no_grad(): # genera traduccion
            generado = mt.generate(**entrada, **opciones)
        salida = tok.batch_decode(generado, skip_special_tokens=True) # de Ids a Texto
        for i, texto in zip(indices, salida):
            traducidas[i] = texto # re ordenar textos
    return traducidas


def traducir_resenas(textos, traductor, **opciones):
    """
    Traduce una lista de reseñas completas.

    Junta las oraciones de todas las reseñas para traducirlas en lotes
    compartidos y después las reagrupa por reseña. Devuelve una lista de
    textos en español, uno por reseña.
    """
    oraciones, pertenencia = [], []
    for n, texto in enumerate(textos):
        partes = dividir_oraciones(texto)
        oraciones.extend(partes)
        pertenencia.extend([n] * len(partes))

    traducidas = traducir_oraciones(oraciones, traductor, **opciones)

    por_resena = [[] for _ in textos]
    for n, oracion in zip(pertenencia, traducidas):
        por_resena[n].append(oracion)
    return [" ".join(partes) for partes in por_resena]


def traducir_corpus(df, traductor, ruta_parcial, col_id="id",
                    col_texto="texto_en", resenas_por_bloque=20, **opciones):
    """
    Devuelve un DataFrame con ``col_id`` y ``texto_es`` para todo el corpus.
    """
    from tqdm.auto import tqdm

    ruta_parcial = Path(ruta_parcial)
    hechas = set()
    if ruta_parcial.exists():
        hechas = set(pd.read_csv(ruta_parcial)[col_id])
        print(f"Reanudando: {len(hechas)} reseñas ya traducidas en "
              f"{ruta_parcial.name}")

    pendientes = df[~df[col_id].isin(hechas)]
    inicio = time.time()
    with tqdm(total=len(df), initial=len(hechas), unit="reseña") as barra:
        for i in range(0, len(pendientes), resenas_por_bloque):
            bloque = pendientes.iloc[i:i + resenas_por_bloque]
            textos_es = traducir_resenas(bloque[col_texto].tolist(),
                                         traductor, **opciones)
            pd.DataFrame({col_id: bloque[col_id].values,
                          "texto_es": textos_es}).to_csv(
                ruta_parcial, mode="a", index=False,
                header=not ruta_parcial.exists(), encoding="utf-8")
            barra.update(len(bloque))

    if len(pendientes):
        minutos = (time.time() - inicio) / 60
        print(f"{len(pendientes)} reseñas traducidas en {minutos:.1f} min")
    return pd.read_csv(ruta_parcial)


# ---------------------------------------------------------------------------
# Etapa 2 · Limpieza

# Palabras funcionales muy frecuentes de cada idioma. Sirven para detectar
# traducciones fallidas: un texto en español casi no contiene las inglesas.
PALABRAS_EN = set("the and is of to it this that was with but for you are "
                  "have not be movie film".split())
PALABRAS_ES = set("el la de que y en los las un una es por con no se lo "
                  "película".split())

# Puntos de corte de la valoración (sección 5.2 del enunciado).
VALORACIONES = ["negativa", "intermedia", "positiva"]


def limpiar_texto(texto):
    """
    Normaliza un texto sin alterar su contenido lingüístico.

    * Quita HTML
    * Colapsa espacios repetidos y saltos de línea.

    """
    if not isinstance(texto, str):
        return ""
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"(?<!\S)[^\w\s](?:\s+[^\w\s]){3,}(?!\S)", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def proporcion_ingles(texto):
    """
    Proporción de palabras funcionales inglesas sobre el total de palabras
    funcionales (inglesas + españolas) encontradas en el texto.

    Vale cerca de 0 en un texto bien traducido y cerca de 1 en uno que quedó
    en inglés. Los títulos de películas sin traducir ("The Land Before Time")
    solo la suben ligeramente.
    """
    palabras = re.findall(r"[a-záéíóúüñ]+", str(texto).lower())
    en = sum(p in PALABRAS_EN for p in palabras)
    es = sum(p in PALABRAS_ES for p in palabras)
    return en / (en + es) if en + es else 1.0


def asignar_valoracion(calificacion):
    """
    Agrupa la calificación de 1 a 10 en tres valoraciones:
    negativa (1-4), intermedia (5-6) y positiva (7-10).
    """
    return pd.cut(calificacion, bins=[0, 4, 6, 10], labels=VALORACIONES)


def limpiar_corpus(df, col_en="texto_en", col_es="texto_es",
                   umbral_ingles=0.5):
    """
    Aplica la limpieza de la etapa 2

    Pasos

    1. Normalización de ``col_en`` y ``col_es`` con :func:`limpiar_texto`.
    2. Eliminación de reseñas cuyo texto en español quedó vacío.
    3. Eliminación de duplicados
    4. Eliminación de traducciones fallidas
    5. Columnas nuevas: ``n_palabras_es`` y ``valoracion``.

    """
    df = df.copy()
    bitacora = []

    def registrar(paso, mascara_eliminar):
        nonlocal df
        bitacora.append({"paso": paso,
                         "eliminadas": int(mascara_eliminar.sum()),
                         "ids": df.loc[mascara_eliminar, "id"].tolist()})
        df = df[~mascara_eliminar]

    df[col_en] = df[col_en].map(limpiar_texto)
    df[col_es] = df[col_es].map(limpiar_texto)

    registrar("Texto en español vacío", df[col_es].str.len() == 0)
    registrar("Texto en español duplicado", df.duplicated(col_es))
    registrar("Misma película y usuario",
              df.duplicated(["movie_id", "user_id"]))
    registrar("Traducción fallida (quedó en inglés)",
              df[col_es].map(proporcion_ingles) > umbral_ingles)

    df["n_palabras_es"] = df[col_es].str.split().str.len()
    df["valoracion"] = asignar_valoracion(df["calificacion"])

    bitacora = pd.DataFrame(bitacora)
    return df.reset_index(drop=True), bitacora
