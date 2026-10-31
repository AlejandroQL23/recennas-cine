# Esquema de MongoDB

Documentación del esquema documental donde se guarda el corpus de reseñas del Proyecto 2.

## 1. Base de datos y colección

| Elemento | Valor |
|---|---|
| Motor | MongoDB Community 7.0 (local) |
| Conexión | `mongodb://localhost:27017` (se puede cambiar con la variable de entorno `MONGO_URI`) |
| Base de datos | `resennas_cine` |
| Colección | `resennas` (un documento por reseña) |
| Código | `src/db_manager.py` · `notebooks/05_migracion_mongodb.ipynb` |

Se usa **una sola colección**. Las reseñas traducidas del Proyecto 1 y las reseñas nativas del scraping comparten el mismo esquema. Los campos que no aplican a un origen quedan en `null`; por ejemplo, `texto_en` en una reseña nativa.

## 2. Estructura del documento

```json
{
  "_id": ObjectId,
  "id_resena": 1,
  "texto_es": "Contenido completo de la reseña en español...",
  "texto_en": "Texto original en inglés (solo reseñas traducidas)",
  "origen_texto": "traducido" | "nativo",
  "titulo_obra": "El laberinto del fauno",
  "tipo_obra": "pelicula" | "serie",
  "generos": ["Drama", "Fantasía"],
  "genero_principal": "Drama",
  "anio_estreno": 2006,
  "calificacion": 9,
  "polaridad": "positiva" | "negativa" | "intermedia",
  "periodo": "antes_2010" | "2010_en_adelante",
  "fuente": "imdb_kaggle" | "filmaffinity" | "sensacine" | "...",
  "url_fuente": "https://...",
  "fecha_resena": ISODate,
  "fecha_recopilacion": ISODate,
  "pos_tags": {
    "nltk":  [ { "oracion": 0, "token": "Lo", "upos": "DET" }, ... ],
    "spacy": [ { "oracion": 0, "token": "Lo", "lema": "él", "upos": "PRON",
                 "tag": "PRON", "morf": "Case=Acc|Gender=Masc|..." }, ... ]
  },
  "embeddings": {
    "word2vec_avg": [ ... ],
    "beto_cls":     [ ... ]
  },
  "metricas": {
    "num_palabras": 181,
    "densidad_lexica": 49.72,
    "ratio_verbos_sustantivos": 0.54,
    "indice_adjetivacion": 12.15,
    "indice_negacion": 1.66
  }
}
```

## 3. Descripción de los campos

| Campo | Tipo | Descripción | Reseñas traducidas (Proyecto 1) | Reseñas nativas (scraping) |
|---|---|---|---|---|
| `_id` | ObjectId | Identificador interno de MongoDB | Automático | Automático |
| `id_resena` | int | Identificador de la reseña en los archivos del proyecto. Sirve para unir el documento con los POS, las métricas y los embeddings. Es único. | `id` de `resenas_limpias.csv` | `id_resena` de `resenas_scraping.csv` |
| `texto_es` | string | Texto de la reseña en español | Traducción con MarianMT, ya limpia | Texto original |
| `texto_en` | string \| null | Texto original en inglés | `texto_en` | `null` |
| `origen_texto` | string | `traducido` o `nativo` | `traducido` | `nativo` |
| `titulo_obra` | string \| null | Título de la película o serie | `null` (el dataset de Kaggle no lo incluye) | Del sitio |
| `tipo_obra` | string | `pelicula` o `serie` | `pelicula` | Del sitio |
| `generos` | array \| null | Lista completa de géneros de la obra | `null` (solo se conservó el género asignado) | Del sitio |
| `genero_principal` | string | Género usado en el análisis, en español | `genero` traducido (ver sección 4) | Primer género del sitio |
| `anio_estreno` | int \| null | Año de estreno de la obra | Año de `release_date` | Del sitio |
| `calificacion` | int | Calificación normalizada a la escala 1-10 | `calificacion` | Del sitio; las escalas de 5 estrellas se multiplican por 2 |
| `polaridad` | string | `negativa` (1-4), `intermedia` (5-6) o `positiva` (7-10) | `valoracion` | Calculada a partir de `calificacion` |
| `periodo` | string | `antes_2010` o `2010_en_adelante`, según el año de la reseña | `periodo` (ver sección 4) | Calculado a partir de `fecha_resena` |
| `fuente` | string | Origen del dato | `imdb_kaggle` | Nombre del sitio |
| `url_fuente` | string | Página de la obra o de la reseña | `https://www.imdb.com/title/<movie_id>/` | URL scrapeada |
| `fecha_resena` | date | Fecha en que se escribió la reseña | `fecha` | Del sitio |
| `fecha_recopilacion` | date \| null | Fecha en que se obtuvo la reseña | `null` | Fecha del scraping |
| `pos_tags.nltk` | array \| null | Un elemento por token: oración, token y etiqueta UPOS de NLTK (cess_esp) | `pos_analysis.csv` | `pos_analysis.csv`, si se etiquetan |
| `pos_tags.spacy` | array \| null | Un elemento por token: oración, token, lema, UPOS, tag y rasgos morfológicos de spaCy | `pos_analysis.csv` | `pos_analysis.csv`, si se etiquetan |
| `embeddings.word2vec_avg` | array \| null | Promedio de los vectores Word2Vec de la reseña | `embeddings_word2vec.npz` | `embeddings_word2vec.npz` |
| `embeddings.beto_cls` | array \| null | Vector [CLS] de BETO (768 dimensiones) | `embeddings_beto.npz` | `embeddings_beto.npz` |
| `metricas.num_palabras` | int | Palabras del texto en español | `n_palabras_es` | Contadas sobre `texto_es` |
| `metricas.densidad_lexica` | float \| null | (NOUN + VERB + ADJ + ADV) / tokens, **en %** | `metricas_por_resena.csv` | `metricas_por_resena.csv` |
| `metricas.ratio_verbos_sustantivos` | float \| null | VERB / NOUN | `metricas_por_resena.csv` | `metricas_por_resena.csv` |
| `metricas.indice_adjetivacion` | float \| null | Adjetivos por cada 100 tokens | `idx_adjetivos` | `idx_adjetivos` |
| `metricas.indice_negacion` | float \| null | Negaciones por cada 100 tokens | `idx_negacion` | `idx_negacion` |

## 4. Transformaciones de los datos del Proyecto 1

**Géneros.** El Proyecto 1 usa los géneros de IMDb en inglés. Se traducen al español para que coincidan con los sitios de scraping en español:

| Proyecto 1 | MongoDB |
|---|---|
| Action | Acción |
| Animation | Animación |
| Comedy | Comedia |
| Drama | Drama |
| Horror | Terror |

**Periodos.**

| Proyecto 1 | MongoDB |
|---|---|
| 1998-2009 | `antes_2010` |
| 2010-2018 | `2010_en_adelante` |

**Polaridad.** Es la columna `valoracion` del Proyecto 1, con los mismos cortes: 1-4 negativa, 5-6 intermedia y 7-10 positiva.

`resenas_scraping.csv`:** usa los nombres de campo del esquema (`id_resena`, `texto_es`, `titulo_obra`, `tipo_obra`, `generos`, `genero_principal`, `anio_estreno`, `calificacion`, `fuente`, `url_fuente`, `fecha_resena`, `fecha_recopilacion`). Los géneros van separados por `|`, y los `id_resena` no deben repetirse con los del Proyecto 1.

## 5. Proceso de carga

`db_manager.migrar()` **reconstruye la colección completa** en cada ejecución:

1. Lee todos los archivos disponibles.
2. Construye un documento por reseña.
3. Borra la colección `resennas`.
4. Inserta los documentos en lotes de 500.
5. Crea los índices.

Así, cuando se generen los embeddings o las reseñas de scraping, basta con guardarlos en los archivos de la sección 5 y volver a ejecutar `notebooks/05_migracion_mongodb.ipynb`. Como la colección se borra en cada carga, el scraper debe guardar sus resultados en `resenas_scraping.csv` además de en MongoDB.

