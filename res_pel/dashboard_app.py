"""
Dashboard interactivo con Plotly Dash.

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

# 2. Diseño de la página: dos filtros y tres gráficos
opciones_genero = [{"label": "Todos", "value": "Todos"}] + [
    {"label": vz.GENEROS_ES[g], "value": g} for g in sorted(resenas["genero"].unique())]
opciones_valoracion = ["Todas", "negativa", "intermedia", "positiva"]

app = Dash(__name__)
app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "20px"}, children=[
    html.H1("Reseñas de cine: análisis morfosintáctico"),
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


# 3. Callback: se ejecuta cada vez que cambia uno de los dos filtros
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


if __name__ == "__main__":
    app.run(debug=False)
