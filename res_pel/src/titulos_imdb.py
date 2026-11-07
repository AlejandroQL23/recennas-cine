"""Obtiene el título de las películas del Proyecto 1 a partir de su id de IMDb.

El dataset de Kaggle no trae el título de las películas, pero sí su id de
IMDb (movie_id, por ejemplo tt0137523). Wikidata, la base de datos abierta de
Wikipedia, guarda el id de IMDb de cada película (propiedad P345), así que se
consulta su servicio SPARQL público para obtener el título original en inglés
(etiqueta en inglés; si no tiene, el título oficial de la obra, P1476).

Los títulos se guardan en data/processed/titulos_imdb.csv (movie_id, titulo),
que luego usa src/unir_corpus.py para llenar titulo_obra. Si el archivo ya
existe, solo se consultan las películas que faltan.

Uso:
    python src/titulos_imdb.py
"""
import time
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
RUTA_RESENAS = RAIZ / "data" / "processed" / "resenas_es.csv"
RUTA_TITULOS = RAIZ / "data" / "processed" / "titulos_imdb.csv"

URL_WIKIDATA = "https://query.wikidata.org/sparql"
USER_AGENT = "CUC-DataMining-Bot/1.0 (Educational Purpose; curso Mineria de Textos)"
LOTE = 100      # ids por consulta
PAUSA = 2.0     # segundos entre consultas

# Películas que Wikidata no tiene asociadas a su id de IMDb. El título se
# infirió de los datos del dataset de Kaggle (IMDB_movie_details.json).
TITULOS_MANUALES = {
    # trama: 'The Bride', miembro del Deadly Viper Assassination Squad liderado
    # por 'Bill'; estreno 2003-10-10; géneros Action, Crime, Thriller
    "tt0266697": "Kill Bill: Volume 1",
}

CONSULTA = """
SELECT ?imdb ?en ?titulo WHERE {{
  VALUES ?imdb {{ {ids} }}
  ?pelicula wdt:P345 ?imdb .
  OPTIONAL {{ ?pelicula rdfs:label ?en . FILTER(LANG(?en) = "en") }}
  OPTIONAL {{ ?pelicula wdt:P1476 ?titulo . }}
}}
"""


def consultar(ids, sesion):
    """Devuelve {movie_id: título} para un lote de ids de IMDb."""
    consulta = CONSULTA.format(ids=" ".join(f'"{i}"' for i in ids))
    for intento in range(3):
        resp = sesion.post(URL_WIKIDATA, data={"query": consulta, "format": "json"}, timeout=60)
        if resp.status_code == 200:
            break
        print(f"  Wikidata respondió {resp.status_code}; reintento en {10 * (intento + 1)} s")
        time.sleep(10 * (intento + 1))
    resp.raise_for_status()

    titulos = {}
    for fila in resp.json()["results"]["bindings"]:
        imdb = fila["imdb"]["value"]
        titulo = fila.get("en", fila.get("titulo", {})).get("value")
        if titulo and imdb not in titulos:
            titulos[imdb] = titulo
    return titulos


def main():
    ids = sorted(pd.read_csv(RUTA_RESENAS, usecols=["movie_id"])["movie_id"].unique())
    previos = pd.read_csv(RUTA_TITULOS) if RUTA_TITULOS.exists() else \
        pd.DataFrame(columns=["movie_id", "titulo"])
    faltan = [i for i in ids if i not in set(previos["movie_id"])]
    print(f"{len(ids)} películas · ya con título: {len(ids) - len(faltan)} · por consultar: {len(faltan)}")

    nuevos = {}
    with requests.Session() as sesion:
        sesion.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
        for inicio in range(0, len(faltan), LOTE):
            lote = faltan[inicio:inicio + LOTE]
            nuevos.update(consultar(lote, sesion))
            print(f"  {min(inicio + LOTE, len(faltan))}/{len(faltan)} consultadas · encontradas {len(nuevos)}")
            time.sleep(PAUSA)

    for movie_id, titulo in TITULOS_MANUALES.items():
        if movie_id in faltan and movie_id not in nuevos:
            nuevos[movie_id] = titulo

    titulos = pd.concat([previos, pd.DataFrame(nuevos.items(), columns=["movie_id", "titulo"])],
                        ignore_index=True).sort_values("movie_id")
    titulos.to_csv(RUTA_TITULOS, index=False, encoding="utf-8")
    sin_titulo = sorted(set(ids) - set(titulos["movie_id"]))
    print(f"Guardado {RUTA_TITULOS.relative_to(RAIZ)}: {len(titulos)} títulos · sin título: {len(sin_titulo)}")
    if sin_titulo:
        print("Sin título en Wikidata:", sin_titulo)


if __name__ == "__main__":
    main()
