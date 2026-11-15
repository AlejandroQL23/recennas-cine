"""
Dashboard interactivo con Plotly Dash.

Pestaña 1: análisis morfosintáctico. Pestaña 2: comparación de representaciones
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

# 2. Estilo: colores y formato que se repiten en toda la página
AZUL_OSCURO = "#1f2a44"
ACENTO = "#e8554e"
FONDO = "#f4f6f9"
COLORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# Todos los gráficos con fondo blanco y la misma paleta de colores
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORES

# Recuadro blanco con sombra para agrupar filtros y gráficos
TARJETA = {"backgroundColor": "white", "borderRadius": "10px", "padding": "15px",
           "marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.1)"}
TITULO_SECCION = {"color": AZUL_OSCURO, "borderLeft": f"5px solid {ACENTO}",
                  "paddingLeft": "10px", "marginTop": "0"}
PESTANA = {"padding": "12px", "fontWeight": "bold"}
PESTANA_ACTIVA = {"padding": "12px", "fontWeight": "bold", "color": "white",
                  "backgroundColor": AZUL_OSCURO, "borderTop": f"3px solid {ACENTO}"}

# 3. Diseño de la página
opciones_genero = [{"label": "Todos", "value": "Todos"}] + [
    {"label": vz.GENEROS_ES[g], "value": g} for g in sorted(resenas["genero"].unique())]
opciones_valoracion = ["Todas", "negativa", "intermedia", "positiva"]
opciones_color = [{"label": "Género", "value": "genero_es"},
                  {"label": "Polaridad", "value": "polaridad"},
                  {"label": "Origen", "value": "origen"}]

# Pestaña 1: dos filtros y tres gráficos
pestana_1 = html.Div(style={"paddingTop": "20px"}, children=[
    html.Div(style=TARJETA, children=[
        html.H3("Filtros", style=TITULO_SECCION),
        html.Div(style={"display": "flex", "gap": "20px", "width": "600px"}, children=[
            html.Div(style={"flex": 1}, children=[
                html.Label("Género"),
                dcc.Dropdown(opciones_genero, "Todos", id="genero", clearable=False)]),
            html.Div(style={"flex": 1}, children=[
                html.Label("Valoración"),
                dcc.Dropdown(opciones_valoracion, "Todas", id="valoracion", clearable=False)]),
        ]),
        html.H4(id="resumen", style={"color": ACENTO, "marginBottom": "0"}),
    ]),
    html.Div(style=TARJETA, children=[
        html.Div(style={"display": "flex"}, children=[
            dcc.Graph(id="grafico_pos", style={"flex": 1}),
            dcc.Graph(id="grafico_metricas", style={"flex": 1}),
            dcc.Graph(id="grafico_adjetivos", style={"flex": 1}),
        ]),
    ]),
])

# Pestaña 2: clasificación y clustering (gráficos fijos) y t-SNE con dos filtros
fig_clasificacion = px.bar(clasificacion, x="tarea", y="accuracy", color="representacion",
                           barmode="group", text_auto=".2f", range_y=[0, 1],
                           title="Accuracy por tarea (Logistic Regression)",
                           labels={"tarea": "Tarea", "accuracy": "Accuracy",
                                   "representacion": "Representación"})
fig_clustering = px.bar(clustering, x="categoria", y="silhouette", color="representacion",
                        barmode="group", text_auto=".3f",
                        title="Silhouette Score (K-Means)",
                        labels={"categoria": "Categoría", "silhouette": "Silhouette",
                                "representacion": "Representación"})

pestana_2 = html.Div(style={"paddingTop": "20px"}, children=[
    html.Div(style={"display": "flex", "gap": "20px"}, children=[
        html.Div(style={**TARJETA, "flex": 1}, children=[
            html.H3("Clasificación", style=TITULO_SECCION),
            dcc.Graph(figure=fig_clasificacion)]),
        html.Div(style={**TARJETA, "flex": 1}, children=[
            html.H3("Clustering", style=TITULO_SECCION),
            dcc.Graph(figure=fig_clustering)]),
    ]),
    html.Div(style=TARJETA, children=[
        html.H3("Visualización t-SNE", style=TITULO_SECCION),
        html.Div(style={"display": "flex", "gap": "20px", "width": "600px"}, children=[
            html.Div(style={"flex": 1}, children=[
                html.Label("Representación"),
                dcc.Dropdown(sorted(tsne["representacion"].unique()), "BETO [CLS]",
                             id="representacion", clearable=False)]),
            html.Div(style={"flex": 1}, children=[
                html.Label("Colorear por"),
                dcc.Dropdown(opciones_color, "genero_es", id="color", clearable=False)]),
        ]),
        dcc.Graph(id="grafico_tsne", style={"height": "600px"}),
    ]),
])

app = Dash(__name__)
app.layout = html.Div(style={"fontFamily": "Segoe UI, sans-serif", "backgroundColor": FONDO,
                             "minHeight": "100vh"}, children=[
    # Encabezado
    html.Div(style={"backgroundColor": AZUL_OSCURO, "color": "white", "padding": "20px 30px"}, children=[
        html.H1("Reseñas de cine", style={"margin": "0"}),
        html.P("Análisis morfosintáctico y comparación de representaciones semánticas",
               style={"margin": "5px 0 0 0", "color": "#c9d1e3"}),
    ]),
    # Pestañas
    html.Div(style={"padding": "20px 30px"}, children=[
        dcc.Tabs([
            dcc.Tab(label="Análisis morfosintáctico", children=pestana_1,
                    style=PESTANA, selected_style=PESTANA_ACTIVA),
            dcc.Tab(label="Comparación de representaciones", children=pestana_2,
                    style=PESTANA, selected_style=PESTANA_ACTIVA),
        ]),
    ]),
])


# 4. Callbacks: se ejecutan cada vez que cambia un filtro
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
                      title=f"t-SNE de las reseñas con {representacion}",
                      labels={"genero_es": "Género", "polaridad": "Polaridad", "origen": "Origen"})


if __name__ == "__main__":
    app.run(debug=False)
