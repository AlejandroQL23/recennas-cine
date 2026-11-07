"""Une las reseñas traducidas (resenas_es.csv) con las de scraping (resenas_scraping.csv).

Guarda data/processed/resenas_combinadas.csv con las columnas de
resenas_scraping.csv: las 14 de resenas_es.csv, en el mismo orden, y al final
las del esquema de MongoDB. En las reseñas traducidas esas columnas extra se
llenan con lo que se sabe de ellas (fuente IMDb/Kaggle, película y su título,
tomado de titulos_imdb.csv) y el resto queda vacío. La columna fuente distingue el origen: imdb_kaggle (traducidas)
o sensacine (nativas).

Uso:
    python src/unir_corpus.py
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROCESADOS = RAIZ / "data" / "processed"
RUTA_TRADUCIDAS = DIR_PROCESADOS / "resenas_es.csv"
RUTA_SCRAPING = DIR_PROCESADOS / "resenas_scraping.csv"
RUTA_TITULOS = DIR_PROCESADOS / "titulos_imdb.csv"
RUTA_SALIDA = DIR_PROCESADOS / "resenas_combinadas.csv"


def caracter_cp1252(m):
    """
    Los textos de IMDb traen caracteres de control U+0080-U+009F que en realidad
    son símbolos de Windows-1252 mal decodificados (U+0085 = '…', U+0096 = '–',
    U+0097 = '—', U+0091/U+0092 = comillas). U+0085 además cuenta como salto de
    línea para algunos programas. Se reemplazan por el símbolo correcto, o se
    eliminan si no tienen equivalente.
    """
    return bytes([ord(m.group())]).decode("cp1252", errors="ignore")


def normalizar(texto):
    """Corrige los caracteres de Windows-1252 y deja el texto en una sola línea."""
    texto = re.sub(r"[\x80-\x9f]", caracter_cp1252, texto)
    return re.sub(r"\s*[\r\n\t\x0b\x0c  ]+\s*", " ", texto)


def main():
    traducidas = pd.read_csv(RUTA_TRADUCIDAS)
    scraping = pd.read_csv(RUTA_SCRAPING)
    assert list(scraping.columns[:len(traducidas.columns)]) == list(traducidas.columns), \
        "Las primeras columnas de resenas_scraping.csv deben ser las de resenas_es.csv"

    # columnas del esquema que se conocen para las reseñas del Proyecto 1
    traducidas["tipo_obra"] = "pelicula"
    traducidas["fuente"] = "imdb_kaggle"
    traducidas["url_fuente"] = "https://www.imdb.com/title/" + traducidas["movie_id"] + "/"
    # título de la película según su id de IMDb (lo genera src/titulos_imdb.py)
    if RUTA_TITULOS.exists():
        titulos = pd.read_csv(RUTA_TITULOS).set_index("movie_id")["titulo"]
        traducidas["titulo_obra"] = traducidas["movie_id"].map(titulos)
        print(f"Reseñas traducidas con título: {traducidas['titulo_obra'].notna().sum()} de {len(traducidas)}")
    else:
        print("No existe titulos_imdb.csv: ejecutar primero src/titulos_imdb.py para llenar titulo_obra")

    combinadas = pd.concat([traducidas, scraping], ignore_index=True)[scraping.columns]
    combinadas = combinadas.sort_values("id", ignore_index=True)

    # Cada reseña debe ocupar una sola línea del CSV; si no, visores como el de
    # PyCharm parten la fila y muestran columnas vacías de más.
    combinadas = combinadas.map(lambda x: normalizar(x) if isinstance(x, str) else x)

    assert combinadas["id"].is_unique, "Hay id repetidos entre los dos archivos"
    # Solo se validan duplicados entre archivos; los que haya dentro de un mismo
    # archivo los elimina la limpieza del notebook 01 ("Texto en español duplicado").
    assert not set(traducidas["texto_es"]) & set(scraping["texto_es"]), \
        "Hay reseñas repetidas entre los dos archivos"

    combinadas.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
    print(f"Guardado {RUTA_SALIDA.relative_to(RAIZ)}: {combinadas.shape[0]} filas, "
          f"{combinadas.shape[1]} columnas")
    print(pd.crosstab(combinadas["genero"], combinadas["fuente"], margins=True))


if __name__ == "__main__":
    main()
