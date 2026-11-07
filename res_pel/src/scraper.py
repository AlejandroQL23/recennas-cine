"""
Web scraping de reseñas de SensaCine
------------------------------------
Extrae críticas de espectadores escritas originalmente en español desde
https://www.sensacine.com y las guarda en data/processed/resenas_scraping.csv.

"""

import base64
import json
import random
import re
import time
import urllib.robotparser
from datetime import date, datetime
from functools import wraps
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parent.parent
RUTA_SALIDA = RAIZ / "data" / "processed" / "resenas_scraping.csv"
RUTA_ESTADO = RAIZ / "data" / "processed" / "scraping_estado.json"
RUTA_RESENAS_ES = RAIZ / "data" / "processed" / "resenas_es.csv"

BASE_URL = "https://www.sensacine.com"
USER_AGENT = "CUC-DataMining-Bot/1.0 (Educational Purpose; curso Mineria de Textos)"

# Columnas de resenas_es.csv (mismo orden) + columnas extra del esquema de MongoDB
COLUMNAS_RESENAS_ES = ["id", "movie_id", "user_id", "fecha", "calificacion", "spoiler",
                       "resumen_en", "texto_en", "texto_es", "n_palabras", "genero",
                       "release_date", "anio", "periodo"]
COLUMNAS_EXTRA = ["titulo_obra", "tipo_obra", "generos", "fuente", "url_fuente",
                  "fecha_recopilacion"]
COLUMNAS = COLUMNAS_RESENAS_ES + COLUMNAS_EXTRA

# Géneros del Proyecto 1: id del género en SensaCine y nombre en resenas_es
GENEROS_SENSACINE = {
    "Action": 13025,     # Acción
    "Animation": 13026,  # Animación
    "Comedy": 13005,     # Comedia
    "Drama": 13008,      # Drama
    "Horror": 13009,     # Terror
}
# Nombre del género en SensaCine -> nombre en resenas_es
GENEROS_ES_A_EN = {
    "Acción": "Action",
    "Animación": "Animation",
    "Dibujos animados": "Animation",
    "Comedia": "Comedy",
    "Drama": "Drama",
    "Terror": "Horror",
}

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}
MESES_EN = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------------------
# Buenas prácticas: robots.txt y reintentos

def check_robots(url_base=BASE_URL, rutas=None, user_agent=USER_AGENT):
    """
    Verifica según robots.txt si las rutas que usa el scraper están permitidas.
    """
    rutas = rutas or ["/peliculas/todas-peliculas/genero-13009/decada-2000/",
                      "/peliculas/pelicula-10568/criticas-espectadores/"]
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{url_base}/robots.txt")
    rp.read()
    return {ruta: rp.can_fetch(user_agent, url_base + ruta) for ruta in rutas}


def retry_on_failure(max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    """Decorador para reintentar requests fallidos con backoff exponencial."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for intento in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if intento == max_retries - 1:
                        raise
                    espera = backoff_factor ** intento + random.uniform(0, 1)
                    print(f"  Error (intento {intento + 1}/{max_retries}): {e}")
                    print(f"  Esperando {espera:.1f}s antes de reintentar...")
                    time.sleep(espera)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Utilidades de parsing

def url_ofuscada(tag):
    """
    SensaCine oculta muchos enlaces en una clase CSS que empieza por "ACr":
    es la URL en base64 con "ACr" intercalado. Devuelve la URL absoluta o None.
    """
    if tag is None:
        return None
    if tag.get("href"):
        return BASE_URL + tag["href"] if tag["href"].startswith("/") else tag["href"]
    for clase in tag.get("class", []):
        if clase.startswith("ACr"):
            try:
                ruta = base64.b64decode(clase.replace("ACr", "")).decode("utf-8")
            except ValueError:
                continue
            return BASE_URL + ruta if ruta.startswith("/") else ruta
    return None


def fecha_es(texto):
    """'Publicada el 24 de diciembre de 2012' -> date(2012, 12, 24); None si no se puede."""
    m = re.search(r"(\d{1,2}) de ([a-záéíóú]+) de (\d{4})", texto or "", re.I)
    if not m or m.group(2).lower() not in MESES:
        return None
    return date(int(m.group(3)), MESES[m.group(2).lower()], int(m.group(1)))


def fecha_formato_resenas_es(fecha):
    """date -> '24 December 2012', el mismo formato de la columna fecha de resenas_es."""
    return f"{fecha.day} {MESES_EN[fecha.month - 1]} {fecha.year}"


def periodo_de(anio):
    """Mismas etiquetas del Proyecto 1; desde 2010 todo va al segundo periodo."""
    return "1998-2009" if anio <= 2009 else "2010-2018"


def genero_principal(generos):
    """Primer género de la película que sea uno de los 5 del Proyecto 1 (mismo criterio)."""
    return next((GENEROS_ES_A_EN[g] for g in generos if g in GENEROS_ES_A_EN), None)


def limpiar(texto):
    """Quita espacios repetidos y saltos de línea."""
    return re.sub(r"\s+", " ", texto or "").strip()


# ---------------------------------------------------------------------------
# Scraper

class SensaCineScraper:
    """Scraper de críticas de espectadores de SensaCine por género y década."""

    def __init__(self, delay=2.0, metas_por_genero=400, max_paginas_listado=20,
                 max_paginas_criticas=3, min_palabras=30, max_palabras=400,
                 decadas=("decada-2010", "decada-2000", "decada-2020", "decada-1990"),
                 ruta_salida=RUTA_SALIDA, ruta_estado=RUTA_ESTADO):
        self.delay = delay
        self.metas_por_genero = metas_por_genero
        self.max_paginas_listado = max_paginas_listado
        self.max_paginas_criticas = max_paginas_criticas
        self.min_palabras = min_palabras
        self.max_palabras = max_palabras
        self.decadas = decadas
        self.ruta_salida = Path(ruta_salida)
        self.ruta_estado = Path(ruta_estado)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "es-ES,es;q=0.9",
        })
        self.requests_hechos = 0
        self._cargar_avance()

    # -- avance --------------------------------------------------------------

    def _cargar_avance(self):
        """Lee lo ya scrapeado para continuar sin repetir películas ni reseñas."""
        self.estado = {"peliculas_visitadas": [], "listados_terminados": []}
        if self.ruta_estado.exists():
            self.estado.update(json.loads(self.ruta_estado.read_text(encoding="utf-8")))

        if self.ruta_salida.exists():
            previas = pd.read_csv(self.ruta_salida)
            self.textos = set(previas["texto_es"])
            self.conteo = previas["genero"].value_counts().to_dict()
            self.siguiente_id = int(previas["id"].max()) + 1
        else:
            self.textos, self.conteo = set(), {}
            # los id continúan después de los del Proyecto 1
            ultimo = pd.read_csv(RUTA_RESENAS_ES, usecols=["id"])["id"].max() \
                if RUTA_RESENAS_ES.exists() else 0
            self.siguiente_id = int(ultimo) + 1

    def _guardar_avance(self, filas):
        """Agrega las filas al CSV y actualiza el archivo de estado."""
        if filas:
            pd.DataFrame(filas, columns=COLUMNAS).to_csv(
                self.ruta_salida, mode="a", index=False, encoding="utf-8",
                header=not self.ruta_salida.exists())
        self.ruta_estado.write_text(json.dumps(self.estado, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    def total(self):
        return sum(self.conteo.values())

    def genero_completo(self, genero):
        return self.conteo.get(genero, 0) >= self.metas_por_genero

    # -- requests ------------------------------------------------------------

    @retry_on_failure(max_retries=3, backoff_factor=2,
                      exceptions=(requests.RequestException, ConnectionError))
    def _fetch(self, url):
        """Request con manejo de errores; espera self.delay después de cada uno."""
        try:
            resp = self.session.get(url, timeout=15)
        finally:
            self.requests_hechos += 1
            time.sleep(self.delay)
        if resp.status_code == 403:
            # el sitio bloqueó el acceso: no se intenta evadir (enunciado, sección 2.2)
            raise PermissionError(f"Acceso bloqueado (403): {url}")
        if resp.status_code == 404:
            raise ValueError(f"Página no encontrada: {url}")
        if resp.status_code == 429:
            raise requests.RequestException("Rate limit excedido (429)")
        if resp.status_code >= 500:
            raise requests.RequestException(f"Error del servidor: {resp.status_code}")
        resp.raise_for_status()
        return resp

    def _get_soup(self, url):
        return BeautifulSoup(self._fetch(url).text, "lxml")

    @staticmethod
    def _siguiente(soup):
        """URL de la página siguiente según el botón 'Siguiente' (o None)."""
        nav = soup.select_one("nav.pagination")
        if nav is None:
            return None
        for boton in nav.select(".button-right"):
            if "Siguiente" in boton.get_text() and "button-disabled" not in boton.get("class", []):
                return url_ofuscada(boton)
        return None

    # -- listados de películas -----------------------------------------------

    def peliculas_de_listado(self, soup):
        """Películas de una página de listado que tienen críticas de usuarios."""
        peliculas = []
        for card in soup.select("li.mdl div.entity-card"):
            enlace = card.select_one("a.meta-title-link")
            if enlace is None:
                continue
            m = re.search(r"pelicula-(\d+)", enlace.get("href", ""))
            criticas = next((url_ofuscada(t) for t in card.select(".rating-title")
                             if "Usuarios" in t.get_text()), None)
            if not m or not criticas:
                continue
            info = card.select_one(".meta-body-info")
            generos = [limpiar(g.get_text()) for g in info.select(".dark-grey-link")] if info else []
            estreno = fecha_es(info.select_one(".date").get_text()) \
                if info and info.select_one(".date") else None
            peliculas.append({
                "id_pelicula": m.group(1),
                "titulo": limpiar(enlace.get_text()),
                "generos": generos,
                "estreno": estreno,
                "url_criticas": criticas,
            })
        return peliculas

    # -- críticas de una película --------------------------------------------

    def parsear_criticas(self, soup, pelicula, genero, url):
        """Reseñas válidas de una página de críticas, ya en el formato del CSV."""
        filas = []
        ahora = datetime.now().isoformat(timespec="seconds")
        for card in soup.select("div.review-card"):
            nota = card.select_one(".stareval-note")
            texto = card.select_one(".content-txt")
            fecha = fecha_es(card.select_one(".review-card-meta-date").get_text()) \
                if card.select_one(".review-card-meta-date") else None
            if nota is None or texto is None or fecha is None:
                continue
            estrellas = float(nota.get_text(strip=True).replace(",", "."))
            calificacion = round(estrellas * 2)  # escala de 5 estrellas -> 1-10
            texto_es = limpiar(texto.get_text(" "))
            n_palabras = len(texto_es.split())
            if (calificacion < 1 or not self.min_palabras <= n_palabras <= self.max_palabras
                    or texto_es in self.textos or fecha.year < 1998):
                continue
            self.textos.add(texto_es)
            filas.append({
                "id": None,  # se asigna al guardar
                "movie_id": f"sensacine-{pelicula['id_pelicula']}",
                "user_id": None,          # no se guardan datos del autor
                "fecha": fecha_formato_resenas_es(fecha),
                "calificacion": calificacion,
                "spoiler": card.select_one("[class*=spoiler]") is not None,
                "resumen_en": None,
                "texto_en": None,
                "texto_es": texto_es,
                "n_palabras": n_palabras,
                "genero": genero,
                "release_date": pelicula["estreno"].isoformat() if pelicula["estreno"] else None,
                "anio": fecha.year,
                "periodo": periodo_de(fecha.year),
                "titulo_obra": pelicula["titulo"],
                "tipo_obra": "pelicula",
                "generos": "|".join(pelicula["generos"]),
                "fuente": "sensacine",
                "url_fuente": url,
                "fecha_recopilacion": ahora,
            })
        return filas

    def scrape_pelicula(self, pelicula, genero):
        """
        Lee hasta max_paginas_criticas páginas de críticas de una película, sin
        pasarse de las reseñas que le faltan al género para llegar a su meta.
        """
        faltan = self.metas_por_genero - self.conteo.get(genero, 0)
        filas, url, pagina = [], pelicula["url_criticas"], 0
        while url and pagina < self.max_paginas_criticas and len(filas) < faltan:
            soup = self._get_soup(url)
            filas += self.parsear_criticas(soup, pelicula, genero, url)
            url = self._siguiente(soup)
            pagina += 1
        for fila in filas[faltan:]:
            self.textos.discard(fila["texto_es"])
        filas = filas[:faltan]
        for fila in filas:
            fila["id"] = self.siguiente_id
            self.siguiente_id += 1
        return filas

    # -- recorrido por género ------------------------------------------------

    def scrape_genero(self, genero, max_requests=None):
        """Recorre los listados de un género (por década) hasta llegar a su meta."""
        id_genero = GENEROS_SENSACINE[genero]
        for decada in self.decadas:
            clave = f"{genero}/{decada}"
            if clave in self.estado["listados_terminados"]:
                continue
            url = f"{BASE_URL}/peliculas/todas-peliculas/genero-{id_genero}/{decada}/"
            pagina = 0
            while url and pagina < self.max_paginas_listado:
                if self.genero_completo(genero) or (max_requests and self.requests_hechos >= max_requests):
                    return
                soup = self._get_soup(url)
                for pelicula in self.peliculas_de_listado(soup):
                    if pelicula["id_pelicula"] in self.estado["peliculas_visitadas"]:
                        continue
                    # se aplica el mismo criterio de género del Proyecto 1
                    asignado = genero_principal(pelicula["generos"])
                    if asignado is None or self.genero_completo(asignado):
                        continue
                    filas = self.scrape_pelicula(pelicula, asignado)
                    self.estado["peliculas_visitadas"].append(pelicula["id_pelicula"])
                    for fila in filas:
                        self.conteo[asignado] = self.conteo.get(asignado, 0) + 1
                    self._guardar_avance(filas)
                    print(f"    {pelicula['titulo'][:40]:<40} {asignado:<9} +{len(filas):>2}"
                          f" | total {self.total()}")
                    if self.genero_completo(genero) or (max_requests and self.requests_hechos >= max_requests):
                        return
                url = self._siguiente(soup)
                pagina += 1
            self.estado["listados_terminados"].append(clave)
            self._guardar_avance([])

    def run(self, generos=None, max_requests=None):
        """
        Ejecuta el scraping completo (o hasta max_requests requests, útil para probar).
        Devuelve el DataFrame con todo lo scrapeado hasta el momento.
        """
        generos = generos or list(GENEROS_SENSACINE)
        print("Iniciando SensaCineScraper")
        print(f"Meta por género: {self.metas_por_genero} | ya scrapeadas: {self.total()}\n")
        try:
            for genero in generos:
                if self.genero_completo(genero):
                    print(f"  {genero}: meta cumplida ({self.conteo.get(genero, 0)})")
                    continue
                print(f"  Scrapeando {genero}...")
                self.scrape_genero(genero, max_requests)
                if max_requests and self.requests_hechos >= max_requests:
                    print(f"\nSe alcanzó el límite de {max_requests} requests")
                    break
        except PermissionError as e:
            print(f"\n{e}\nEl sitio bloqueó el acceso: se detiene el scraping (cambiar de fuente).")
        finally:
            self.session.close()

        print(f"\nRequests hechos: {self.requests_hechos} | reseñas totales: {self.total()}")
        return pd.read_csv(self.ruta_salida) if self.ruta_salida.exists() else pd.DataFrame(columns=COLUMNAS)
