"""
POS Tagging con spaCy y NLTK.

* spaCy (es_core_news_md): etiquetado del corpus por lotes, con lema.
* NLTK: etiquetador de n-gramas entrenado con el corpus cess_esp.

"""

import pandas as pd

from traduccion_limpieza import asegurar_recursos_nltk

# ---------------------------------------------------------------------------
# Etapa 3 · POS Tagging

MODELO_SPACY = "es_core_news_md"


def cargar_spacy(modelo=MODELO_SPACY):
    """
    Carga el modelo de spaCy para español sin el reconocimiento de entidades
    (ner), que no se usa y así el etiquetado es más rápido.
    """
    import spacy

    return spacy.load(modelo, disable=["ner"])


def etiquetar_spacy(ids, textos, nlp, batch_size=64):
    """
    Etiqueta las reseñas con spaCy y devuelve un DataFrame con un token por
    fila.

    Las reseñas se procesan por lotes con nlp.pipe, mucho más rápido que
    llamar a nlp reseña por reseña. Los tokens que son solo espacios se
    descartan.
    """
    from tqdm.auto import tqdm # barra de progreso

    ids, textos = list(ids), list(textos)
    filas = []
    docs = nlp.pipe(textos, batch_size=batch_size)
    for id_resena, doc in tqdm(zip(ids, docs), total=len(ids), unit="reseña"):
        for n_oracion, oracion in enumerate(doc.sents):
            for token in oracion:
                if token.is_space:
                    continue
                filas.append({"id": id_resena, "oracion": n_oracion,
                              "token": token.text, "lema": token.lemma_,
                              "upos_spacy": token.pos_,
                              "tag_spacy": token.tag_,
                              "morf": str(token.morph)})
    return pd.DataFrame(filas)


def eagles_a_upos(tag):
    """
    Convierte una etiqueta EAGLES (corpus cess_esp) a Universal POS.

    En EAGLES la primera letra indica la categoría (n sustantivo, v
    verbo, a adjetivo etc) y las siguientes, rasgos como género, número o
    persona: ncms000 es un sustantivo común masculino singular
    """
    t = tag.lower()
    if t.startswith("np"):
        return "PROPN"
    if t.startswith("cs"):
        return "SCONJ"
    if t.startswith(("va", "vs")):
        return "AUX"
    mapa = {"a": "ADJ", "c": "CCONJ", "d": "DET", "f": "PUNCT", "i": "INTJ",
            "n": "NOUN", "p": "PRON", "r": "ADV", "s": "ADP", "v": "VERB",
            "z": "NUM", "w": "NUM"}
    return mapa.get(t[:1], "X")


def oraciones_cess_upos():
    """
    Devuelve las oraciones de cess_esp como listas de (palabra, UPOS).

    Las palabras se pasan a minúsculas para que el etiquetador no distinga
    "Película" de "película"
    """
    from nltk.corpus import cess_esp
    # *0* palabras que se omiten, ya que si no se hace "ensucia" el resultado
    #“Llegamos tarde” → hay un sujeto implícito (“nosotros”) que no se pronuncia
    asegurar_recursos_nltk(("corpora/cess_esp",))
    return [[(w.lower(), eagles_a_upos(t)) for w, t in oracion if w != "*0*"]
            for oracion in cess_esp.tagged_sents()]


def entrenar_etiquetadores(oraciones, proporcion_entreno=0.9):
    """
    Entrena los etiquetadores de n-gramas de NLTK

    * El 90 % de las oraciones se usa para entrenar y el 10 % final
      para test
    * unigrama: asigna a cada palabra su etiqueta más frecuente en el
      entrenamiento; si no la conoce, recurre a DefaultTagger(NOUN)
    * bigrama: considera también la etiqueta de la palabra anterior; si
      ese contexto no se vio en el entrenamiento, recurre al unigrama

    Devuelve un diccionario con los etiquetadores, su exactitud sobre la
    prueba y el vocabulario de entrenamiento.
    """
    from nltk.tag import BigramTagger, DefaultTagger, UnigramTagger

    corte = int(len(oraciones) * proporcion_entreno)
    entreno, prueba = oraciones[:corte], oraciones[corte:]

    defecto = DefaultTagger("NOUN")
    unigrama = UnigramTagger(entreno, backoff=defecto)
    bigrama = BigramTagger(entreno, backoff=unigrama)

    return {
        "unigrama": unigrama,
        "bigrama": bigrama,
        "exactitud": {"Por defecto (todo NOUN)": defecto.accuracy(prueba),
                      "Unigrama": unigrama.accuracy(prueba),
                      "Bigrama": bigrama.accuracy(prueba)},
        "vocabulario": {w for oracion in entreno for w, _ in oracion},
        "n_entreno": len(entreno),
        "n_prueba": len(prueba),
    }


def etiquetar_nltk(tokens, etiquetador):
    """
    Etiqueta con NLTK los MISMOS tokens que produjo spaCy.

    Usar la tokenización de spaCy garantiza que ambas herramientas etiqueten
    exactamente las mismas unidades

    """
    from itertools import groupby

    palabras = tokens["token"].str.lower().tolist()
    claves = zip(tokens["id"], tokens["oracion"])

    oraciones, inicio = [], 0
    for _, grupo in groupby(claves):
        largo = sum(1 for _ in grupo)
        oraciones.append(palabras[inicio:inicio + largo])
        inicio += largo

    return [etiqueta for oracion in etiquetador.tag_sents(oraciones)
            for _, etiqueta in oracion]
