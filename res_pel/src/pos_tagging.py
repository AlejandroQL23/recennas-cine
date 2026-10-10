"""
Funciones reutilizables del pipeline de reseñas de cine.

Etapa 1 · Traducción automática inglés -> español
--------------------------------------------------
Se usa el modelo MarianMT ``Helsinki-NLP/opus-mt-en-es``: es gratuito, corre
localmente (sin claves de API).

* La traducción del corpus completo tarda horas en CPU, así que se guarda un
  archivo parcial cada pocas reseñas. Si el proceso se interrumpe, al volver a
  ejecutarlo continúa donde quedó en lugar de empezar de cero.
"""

import re
import time
from pathlib import Path

import pandas as pd

MODELO_TRADUCCION = "Helsinki-NLP/opus-mt-en-es"


# ---------------------------------------------------------------------------
# Preparación del texto en inglés
# ---------------------------------------------------------------------------

def asegurar_recursos_nltk():
    """Descarga (solo la primera vez) el segmentador de oraciones de NLTK."""
    import nltk

    for recurso in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{recurso}")
        except LookupError:
            nltk.download(recurso, quiet=True)


def preparar_texto_en(texto):
    """
    Normaliza el texto en inglés antes de dividirlo en oraciones.

    * Elimina etiquetas HTML (por ejemplo ``<br />``).
    * Inserta un espacio cuando una oración termina pegada a la siguiente
      (``"impression.This film"`` -> ``"impression. This film"``).
    * Colapsa los espacios repetidos.

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

    Usa ``nltk.sent_tokenize`` y, si alguna "oración" sigue siendo muy larga
    (reseñas sin puntuación), la corta en trozos de ``max_palabras`` palabras
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
# ---------------------------------------------------------------------------

def cargar_traductor(modelo=MODELO_TRADUCCION, dispositivo=None):
    """
    Devuelve un diccionario con ``tok``, ``modelo`` y ``dispositivo``
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
        Resultado de :func:`cargar_traductor`.
    lote : int
        Oraciones por lote. En CPU, 8-16 suele ser lo más eficiente.
    max_length : int
        Máximo de tokens de entrada; lo que sobrepase se trunca.
    num_beams : int | None
        Ancho de la búsqueda en haz. ``None`` usa el valor del modelo (4),
        que da mejor calidad;
    """
    import torch

    tok, mt, dispositivo = (traductor["tok"], traductor["modelo"],
                            traductor["dispositivo"])

    # Se ordenan por longitud para que cada lote tenga oraciones de tamaño
    # parecido (menos relleno = menos cómputo) y luego se restaura el orden.
    orden = sorted(range(len(oraciones)), key=lambda i: len(oraciones[i]))
    traducidas = [""] * len(oraciones)

    opciones = {} if num_beams is None else {"num_beams": num_beams}
    for inicio in range(0, len(orden), lote):
        indices = orden[inicio:inicio + lote]
        entrada = tok([oraciones[i] for i in indices], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_length).to(dispositivo)
        with torch.no_grad():
            generado = mt.generate(**entrada, **opciones)
        salida = tok.batch_decode(generado, skip_special_tokens=True)
        for i, texto in zip(indices, salida):
            traducidas[i] = texto
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
    Cada ``resenas_por_bloque`` reseñas se agregan al CSV ``ruta_parcial``
    (columnas ``col_id`` y ``texto_es``). Si el archivo ya existe, se omiten
    las reseñas que ya contiene, de modo que el proceso puede detenerse
    (interrumpir el kernel, apagar el equipo) y reanudarse sin perder trabajo.

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
