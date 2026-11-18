# Documentación del uso de Inteligencia Artificial

## 1. Uso de IA (Claude)

### 1.1 Extracción del corpus

**Prompt:**
> Te adjunto las instrucciones del contenido del corpus. Además, aquí están las reseñas; podríamos hacer una selección aleatoria para obtener las 3 000 reseñas. El formato debe ser `.csv`.

---

### 1.2 Creación del archivo `.gitignore`

**Prompt:**
> Tengo un proyecto en Python. Dentro de su estructura hay una carpeta `data/raw`, en la cual no se deben cargar los datos originales debido a su peso. ¿Puedes crear un `.gitignore` para evitar que ese archivo, que es un `.csv`, se publique?

---

### 1.3 Expresiones regulares para limpieza de texto

**Uso:** eliminar etiquetas HTML, quitar espacios sobrantes y separar oraciones que quedaron pegadas.

**Prompt:**
> En el desarrollo del proyecto debo realizar una limpieza de los datos. ¿Podrías ayudarme a crear una función en Python con expresiones regulares que me ayude a limpiar todos los datos? Esto es lo que necesito: quitar etiquetas HTML (si las hay), eliminar vacíos y eliminar duplicados.

---

### 1.4 Traducción del corpus

**Nota:** al intentarlo desde Google Colab no funcionó, por lo que se realizó dentro del mismo proyecto.

**Prompt:**
> Ayúdame a crear una función para traducir un corpus que está en inglés. Las indicaciones recomiendan lo siguiente: para la traducción se sugiere el modelo `Helsinki-NLP/opus-mt-en-es` (MarianMT). El CSV resultante debe conservar el texto original en inglés junto al texto en español. Hazlo de la manera más eficiente posible para que el proceso no tarde tanto.

---

### 1.5 Visualización con Plotly Dash

**Prompt:**
> Ayúdame con ejemplos básicos y documentados de Plotly Dash, con gráficos y selectores.

---

### 1.6 Redacción de este documento

**Prompt:**
> Este es el contenido de un archivo `.md` que tiene los usos y prompts hechos para la IA. Corrige la ortografía y dale un formato formal y bonito.

---

## 2. Reflexión sobre el aprendizaje con IA

Aprendí que la IA no reemplaza mi criterio. Varias de sus propuestas iniciales no se ajustaban a lo que pedí, por lo que tuve que revisar cada respuesta contra el enunciado y las indicaciones del proyecto, y corregir el rumbo cuando fue necesario.

Pedirle que explicara línea por línea el código generado, como se hizo con la traducción, sirvió para comprenderlo y poder defenderlo en la demostración, en lugar de copiarlo sin más.

---

## 3. Modificaciones al código y al análisis generados por IA

| Aspecto | Modificación realizada |
|---|---|
| **Periodos temporales** | Se aclaró que las reseñas debían pertenecer a años distintos, por lo que los periodos quedaron sin años compartidos: 1998-2009 y 2010-2018. |
| **Traducción** | El plan original traducía en Google Colab. Se cambió para que la traducción quedara dentro del mismo pipeline local, sin depender de herramientas externas. |
| **Documentación** | Todo el código generado debe estar documentado en español, con comentarios y celdas explicativas en los notebooks. |
| **Imports** | Se añadieron imports en ciertos bloques de código, ya que al ejecutarlos generaban errores. |



### Segunda parte del proyecto

Durante la segunda fase del proyecto, se utilizó nuevamente la IA como herramienta de apoyo para desarrollar los componentes relacionados con MongoDB y la extracción de información de sitios web.

| Aspecto | Uso de IA |
|---|---|
| **Estructura e índices en MongoDB** | Se solicitó asistencia para el diseño y la creación de la estructura de la base de datos en MongoDB, así como para la definición de los índices necesarios, tomando como referencia los requerimientos establecidos en el enunciado del proyecto. |
| **Web Scraping de SensaCine** | Se proporcionó a la IA el documento de instrucciones facilitado por el profesor, con el objetivo de obtener orientación en la implementación de los componentes necesarios para realizar el scraping del sitio web SensaCine. Dicho documento se utilizó como referencia para orientar el desarrollo y garantizar el cumplimiento de los requisitos establecidos. |
| **Mejoras en la interfaz gráfica** | Se utilizó la IA como apoyo para realizar mejoras en la interfaz gráfica de la aplicación, buscando optimizar su presentación visual, organización y experiencia de usuario. |
| **Identificación de títulos de películas** | Se solicitó ayuda para identificar y determinar los títulos correspondientes a las películas asociadas a las 15 reseñas obtenidas previamente, con el propósito de facilitar su correcta identificación y procesamiento dentro del proyecto. |
| **Gráficos utilizando Word2Vec (W2V)** | Se solicitó asistencia en la elaboración y mejora de los gráficos relacionados con los resultados obtenidos mediante Word2Vec (W2V), con el objetivo de facilitar su visualización e interpretación. |
| **Búsquedas semánticas** | Se utilizó la IA como apoyo para el desarrollo y la comprensión de las búsquedas semánticas implementadas en el proyecto, facilitando el análisis de similitud entre textos y la recuperación de información relevante a partir del contenido de las reseñas. |
| **Documentación de archivos Python (.py) y Jupyter Notebook (.ipynb)** | Se solicitó ayuda para documentar los archivos `.py` y `.ipynb` del proyecto, incluyendo la descripción de su propósito, funcionalidades, procesos implementados y principales componentes, con el fin de mejorar la claridad, comprensión y mantenimiento del código. |
| **Adaptación de archivos guía a funciones Python** | Se utilizó la IA como apoyo para trasladar y adaptar el contenido de los archivos guía proporcionados por el profesor a funciones implementadas en archivos `.py`, facilitando su reutilización, organización e integración dentro del proyecto. |







