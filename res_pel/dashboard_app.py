"""
Dashboard interactivo con Plotly Dash.

(BoW vs. Word2Vec vs. BETO), con los resultados que guarda el notebook 09.

abrir http://127.0.0.1:8050 en el navegador
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src"))

import metrics as mt
import visualizations as vz

# 1. Datos: se cargan una sola vez, al iniciar la app
resenas = pd.read_csv(RAIZ / "data" / "results" / "metricas_por_resena.csv")
tokens = mt.sin_puntuacion(mt.cargar_tokens(RAIZ / "data" / "results" / "pos_analysis.csv"))

# Resultados de la comparación (notebook 09)
clasificacion = pd.read_csv(RAIZ / "data" / "results" / "comparacion_clasificacion.csv")
clustering = pd.read_csv(RAIZ / "data" / "results" / "comparacion_clustering.csv")
tsne = pd.read_csv(RAIZ / "data" / "results" / "comparacion_tsne.csv")

# 2. Diseño de la página
opciones_genero = [{"label": "Todos", "value": "Todos"}] + [
    {"label": vz.GENEROS_ES[g], "value": g} for g in sorted(resenas["genero"].unique())]
opciones_valoracion = ["Todas", "negativa", "intermedia", "positiva"]

# Pestaña 1: dos filtros y tres gráficos
pestana_1 = html.Div(children=[
    html.Div(style={"display": "flex", "gap": "20px", "width": "600px"}, children=[
        html.Div(style={"flex": 1}, children=[
            html.Label("Género"),
            dcc.Dropdown(opciones_genero, "Todos", id="genero", clearable=False)]),
        html.Div(style={"flex": 1}, children=[
            html.Label("Valoración"),
            dcc.Dropdown(opciones_valoracion, "Todas", id="valoracion", clearable=False)]),
    ]),
    html.H3(id="resumen"),
    html.Div(style={"display": "flex"}, children=[
        dcc.Graph(id="grafico_pos", style={"flex": 1}),
        dcc.Graph(id="grafico_metricas", style={"flex": 1}),
        dcc.Graph(id="grafico_adjetivos", style={"flex": 1}),
    ]),
])

# Pestaña 2: clasificación y clustering (gráficos fijos) y t-SNE con dos filtros
fig_clasificacion = px.bar(clasificacion, x="tarea", y="accuracy", color="representacion",
                           barmode="group", title="Clasificación: accuracy por tarea")
fig_clustering = px.bar(clustering, x="categoria", y="silhouette", color="representacion",
                        barmode="group", title="Clustering con K-Means: Silhouette Score")

pestana_2 = html.Div(children=[
    html.Div(style={"display": "flex"}, children=[
        dcc.Graph(figure=fig_clasificacion, style={"flex": 1}),
        dcc.Graph(figure=fig_clustering, style={"flex": 1}),
    ]),
    html.Div(style={"display": "flex", "gap": "20px", "width": "600px"}, children=[
        html.Div(style={"flex": 1}, children=[
            html.Label("Representación"),
            dcc.Dropdown(sorted(tsne["representacion"].unique()), "BETO [CLS]",
                         id="representacion", clearable=False)]),
        html.Div(style={"flex": 1}, children=[
            html.Label("Colorear por"),
            dcc.Dropdown(["genero_es", "polaridad", "origen"], "genero_es",
                         id="color", clearable=False)]),
    ]),
    dcc.Graph(id="grafico_tsne", style={"height": "600px"}),
])

app = Dash(__name__)
app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "20px"}, children=[
    html.H1("Reseñas de cine"),
    dcc.Tabs([
        dcc.Tab(label="Análisis morfosintáctico", children=pestana_1),
        dcc.Tab(label="Comparación de representaciones", children=pestana_2),
    ]),
])


# 3. Callbacks: se ejecutan cada vez que cambia un filtro
@app.callback(
    Output("resumen", "children"),
    Output("grafico_pos", "figure"),
    Output("grafico_metricas", "figure"),
    Output("grafico_adjetivos", "figure"),
    Input("genero", "value"),
    Input("valoracion", "value"),
)
def actualizar(genero, valoracion):
    filtro, pos, promedios, top = vz.calcular_filtro(tokens, resenas, genero, valoracion)

    fig_pos = px.bar(x=pos.index, y=pos.values, title="Categorías gramaticales",
                     labels={"x": "Categoría (UPOS)", "y": "% de tokens"})
    fig_metricas = px.bar(x=promedios.index, y=promedios.values, title="Métricas",
                          labels={"x": "", "y": "Por cada 100 tokens"})
    fig_adjetivos = px.bar(x=top.values, y=top.index, orientation="h", title="Adjetivos más frecuentes",
                           labels={"x": "Frecuencia", "y": ""})
    fig_metricas.update_traces(marker_color="#DD8452")
    fig_adjetivos.update_traces(marker_color="#55A868")

    resumen = f"{len(filtro)} reseñas seleccionadas"
    return resumen, fig_pos, fig_metricas, fig_adjetivos


@app.callback(
    Output("grafico_tsne", "figure"),
    Input("representacion", "value"),
    Input("color", "value"),
)
def actualizar_tsne(representacion, color):
    datos = tsne[tsne["representacion"] == representacion]
    return px.scatter(datos, x="x", y="y", color=color, hover_name="titulo_obra", opacity=0.5,
                      title=f"t-SNE de las reseñas con {representacion}")


if __name__ == "__main__":
    app.run(debug=False)
