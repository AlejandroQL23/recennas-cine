"""
Funciones de gráficos

* guardar_figura: guarda un gráfico de matplotlib como PNG en outputs/figures.
* adjetivos_caracteristicos: adjetivos que más distinguen a un grupo de otro.
* calcular_filtro: datos del dashboard para un género y una valoración.
* crear_dashboard: dashboard interactivo de Plotly con filtros por género y
  valoración, exportado como HTML.

La versión con Plotly Dash (app web con servidor) está en dashboard_app.py.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Nombres en español de los géneros, para títulos y leyendas.
GENEROS_ES = {"Action": "Acción", "Animation": "Animación", "Comedy": "Comedia",
              "Drama": "Drama", "Horror": "Terror"}


def guardar_figura(fig, carpeta, nombre):
    """Guarda la figura como PNG en ``carpeta/nombre.png`` y la muestra."""
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()


def adjetivos_caracteristicos(adjetivos, grupo_a, grupo_b, col_grupo="valoracion", n=10):
    """
    Adjetivos más característicos de X vs Y

    Para cada lema se calcula su frecuencia relativa en cada grupo. Los primeros
    son los que A usa mucho más que B. Solo se consideran lemas que aparecen
    al menos 20 veces, para evitar palabras poco usadas
    """
    frecuencia = pd.crosstab(adjetivos["lema"].str.lower(), adjetivos[col_grupo])
    frecuencia = frecuencia[frecuencia.sum(axis=1) >= 20]
    por_100 = frecuencia / frecuencia.sum() * 100
    diferencia = (por_100[grupo_a] - por_100[grupo_b]).sort_values(ascending=False)
    return diferencia.head(n)


METRICAS_DASHBOARD = {"idx_adjetivos": "Adjetivos", "idx_verbos": "Verbos",
                      "idx_negacion": "Negaciones", "idx_pron_1": "Pron. 1.ª persona"}


def calcular_filtro(tokens, resenas, genero="Todos", valoracion="Todas"):
    """
    Datos de los 3 gráficos del dashboard para un género y una valoración
    ("Todos" / "Todas" = sin filtrar). La usan el HTML y la app de Dash.

    Devuelve: reseñas filtradas, % de tokens por categoría gramatical,
    promedio de las métricas y los 10 adjetivos más frecuentes.
    """
    filtro = resenas
    if genero != "Todos":
        filtro = filtro[filtro["genero"] == genero]
    if valoracion != "Todas":
        filtro = filtro[filtro["valoracion"] == valoracion]
    sub_tokens = tokens[tokens["id"].isin(filtro["id"])]

    pos = sub_tokens["upos_spacy"].value_counts(normalize=True).head(10) * 100
    promedios = filtro[list(METRICAS_DASHBOARD)].mean().rename(METRICAS_DASHBOARD)
    adjetivos = sub_tokens.loc[sub_tokens["upos_spacy"] == "ADJ", "lema"].str.lower()
    top = adjetivos.value_counts().head(10).sort_values()
    return filtro, pos, promedios, top


