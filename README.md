# Análisis morfosintáctico de reseñas de cine mediante POS Tagging

Proyecto 1 del curso, impartido por el profesor Osvaldo González Chaves.
Aplicación de NLTK y spaCy al estudio de la opinión cinematográfica en español.

## Autor

| Nombre | Carné     |
|---|-----------|
| Alejandro Quesada | 305230384 |

---

## Descripción

En este proyecto construí un corpus de 3.000 reseñas de películas en español y lo analicé con técnicas de etiquetado de categorías gramaticales (POS Tagging). La idea es ver si la forma en que se escribe una reseña (cuántos adjetivos, verbos, negaciones o pronombres usa, qué tan largas son sus oraciones) cambia según tres variables:

- **el género de la película** (Acción, Animación, Comedia, Drama y Terror),
- **la valoración** que el usuario le dio (negativa, intermedia o positiva),
- **el periodo** en que se escribió la reseña (1998–2009 frente a 2010–2018).

Como no encontré un dataset en español con género, fecha y calificación, partí de reseñas de "IMDB Spoiler Dataset" en inglés y las traduje automáticamente con un modelo local. Para el etiquetado usé dos herramientas: spaCy, con un modelo preentrenado para español, y un etiquetador de NLTK que entrené yo mismo con el corpus `cess_esp`. Además las comparé entre sí.

---

## Dataset

- **Fuente:** IMDB Spoiler Dataset de Kaggle. Trae más de 570 mil reseñas de usuarios de IMDb con texto, calificación de 1 a 10, fecha y marca de spoiler, y un segundo archivo con el género y la fecha de estreno de cada película.
- **Muestra:** 3.000 reseñas

| Requisito del enunciado | Mínimo | Muestra del proyecto |
|---|---|---|
| Reseñas | 3.000 | 3.000 |
| Géneros | 5 con al menos 400 cada uno | 5 con 600 cada uno |
| Periodos | 2 | 1998–2009: 1.500 · 2010–2018: 1.500 |
| Longitud de cada reseña | 30 a 400 palabras | 31 a 400 palabras |


---

## Pipeline

El trabajo se divide en cinco etapas. Cada una guarda su resultado en un archivo, para no tener que repetir los pasos lentos (traducción y etiquetado) cada vez que ajustaba un análisis o un gráfico.

| Etapa | Notebook | Salida |
|---|---|---|
| 1. Carga y traducción | `01_carga_traduccion_limpieza.ipynb` | `data/processed/resenas_es.csv` |
| 2. Limpieza | `01_carga_traduccion_limpieza.ipynb` | `data/processed/resenas_limpias.csv` |
| 3. POS Tagging | `02_pos_tagging.ipynb` | `data/results/pos_analysis.csv` |
| 4. Análisis | `03_analisis_metricas.ipynb` | `data/results/metricas_por_resena.csv`, `respuestas_preguntas.csv`, `revision_manual.csv` |
| 5. Visualización | `04_visualizaciones.ipynb` | `outputs/figures/*.png` |

### 1. Traducción automática

- **Modelo:** [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) (MarianMT). Lo elegí porque es gratuito, corre localmente y no necesita claves de API.
- **Ejecución:** en CPU, sin Google Colab. Las 3.000 reseñas tardaron unas **6 horas** (357,7 minutos).
- **Cómo traduje cada reseña:**
  - El modelo corta las entradas largas, así que dividí cada reseña en oraciones con `nltk.sent_tokenize`, las traduje por lotes y luego las volví a unir.
  - Antes de dividir, separé las oraciones que en IMDb vienen pegadas ("film.This"). Si no, quedaban fragmentos tan largos que el modelo los truncaba.

- **Reanudable:** el avance se guarda cada 20 reseñas en `resenas_es_parcial.csv`, así que si se interrumpía el proceso podía continuar donde había quedado.
- El CSV final conserva el texto original en inglés (`texto_en`) junto a la traducción (`texto_es`). Las traducciones **no se corrigieron a mano**, como pide el enunciado.

### 2. Limpieza

- Quité etiquetas y entidades HTML y normalicé los espacios.
- Revisé reseñas vacías, duplicadas (por texto) y traducciones fallidas, es decir, textos que quedaron en inglés. Las detecté comparando cuántas palabras funcionales inglesas y españolas tiene cada texto. **No hubo que eliminar ninguna reseña**, y se normalizaron 26 textos.
- Creé la columna `valoracion`: **negativa** (1–4), **intermedia** (5–6) y **positiva** (7–10). Quedaron 684, 374 y 1.942 reseñas respectivamente.

### 3. POS Tagging

- **spaCy** (`es_core_news_md`): procesé el corpus por lotes con `nlp.pipe` y guardé un token por fila con su lema, etiqueta UPOS, etiqueta detallada y rasgos morfológicos. 
- **NLTK:** el etiquetador preentrenado de NLTK solo funciona en inglés, así que entrené uno con el corpus `cess_esp`, de unas 6.000 oraciones anotadas con el esquema EAGLES. Convertí sus etiquetas a UPOS para poder compararlo con spaCy y lo apliqué sobre **los mismos tokens** que produjo spaCy.

### 4. Métricas

Las calculé **por reseña** a partir de las etiquetas de spaCy y luego las promedié por grupo. Excluí la puntuación

---

## Resultados por pregunta de investigación

Respondí dos preguntas de cada eje.

| | Pregunta | Respuesta | Datos |
|---|---|---|---|
| **A1** | ¿Las reseñas de Terror usan más adjetivos que las de Comedia? | No | 8,25 vs. 8,58 adjetivos por 100 tokens |
| **A2** | ¿Las reseñas de Acción tienen más verbos que las de Drama? | No, al revés | Acción 10,13 vs. Drama 10,73 |
| **B1** | ¿Las reseñas con calificación baja usan más negaciones? | Sí | Negativas 2,62 vs. positivas 1,70 por 100 tokens |
| **B6** | ¿Las reseñas negativas son más largas o tienen oraciones más largas? | No | 182 vs. 184 tokens; 19,9 vs. 20,3 tokens por oración |
| **C2** | ¿Ha cambiado la proporción de adjetivos con el tiempo? | No | 8,50 (1998–2009) vs. 8,58 (2010–2018), sin tendencia por año |
| **C4** | ¿Cambia con los años la longitud de las reseñas? | Sí | 173,9 → 196,5 tokens  |


---

## Comparación NLTK vs. spaCy

- Las dos herramientas coinciden en el **82,8 %** de los tokens (sin puntuación).
- Coinciden casi siempre en las categorías funcionales (preposiciones, conjunciones, determinantes) y mucho menos en nombres propios, verbos y adjetivos.

**Conclusión:** para este corpus spaCy es claramente más confiable. La mayoría de los errores de NLTK vienen de que `cess_esp` es un corpus de prensa pequeño, sin vocabulario de cine ni nombres propios de este tipo.

---

## Visualizaciones y dashboard

En `outputs/figures/` están los siete gráficos del análisis:

1. Distribución de categorías gramaticales por género
2. Métricas por género (A1, A2)
3. Adjetivos y negaciones según la calificación (B1)
4. Longitud de reseña y de oración por valoración (B6)
5. Evolución por año de adjetivos y longitud (C2, C4)
6. Matriz de confusión NLTK vs. spaCy
7. Adjetivos más característicos por valoración

El dashboard se hace con  **Plotly Dash** 


Ambas versiones muestran, para la selección, la distribución de categorías gramaticales, cuatro métricas y los adjetivos más frecuentes.

---

## Estructura del repositorio

```
res_pel/
├── README.md
├── requirements.txt
├── USO_IA.md                 # Documentación del uso de IA
├── dashboard_app.py             # Dashboard con Plotly Dash
├── data/
│   ├── raw/                     # Dataset original (no se sube)
│   ├── processed/
│   │   ├── resenas_es.csv       # Muestra traducida
│   │   └── resenas_limpias.csv
│   └── results/
│       ├── pos_analysis.csv     # Un token por fila con etiquetas de spaCy y NLTK
│       ├── metricas_por_resena.csv
│       ├── respuestas_preguntas.csv
│       └── revision_manual.csv  # Revisión manual de 50 discrepancias
├── notebooks/
│   ├── 01_carga_traduccion_limpieza.ipynb
│   ├── 02_pos_tagging.ipynb
│   ├── 03_analisis_metricas.ipynb
│   └── 04_visualizaciones.ipynb
├── src/
│   ├── traduccion_limpieza.py   # Funciones de traducción y limpieza
│   ├── pos_tagging.py           # Funciones de etiquetado (spaCy y NLTK)
│   ├── metrics.py               # Funciones de cálculo de métricas y comparación
│   └── visualizations.py        # Funciones de gráficos y dashboard
└── outputs/
    ├── figures/                 # Gráficos PNG
```

El enunciado propone que las funciones de traducción, limpieza y etiquetado vayan juntas en `pos_tagging.py`. Yo las separé en dos archivos (`traduccion_limpieza.py` y `pos_tagging.py`), uno por etapa, porque así cada archivo queda más corto y es más fácil de entender.

---

## Uso de inteligencia artificial

El uso de herramientas de IA durante el proyecto está documentado en USO_IA.md 
