# Misión 2: Predicción de éxito de películas en el cine con Técnicas de Regresión

## Objetivo:
En esta misión, explorarás cómo construir y evaluar modelos de regresión para predecir el éxito de las películas en términos de ingresos y aceptación. Además, reforzarás tus habilidades en el diagnóstico de modelos y aplicarás técnicas básicas de aprendizaje supervisado.

---

## Actividades:

### 1. **Introducción al Aprendizaje Automático y Dataset de Películas**
   - Realiza una breve investigación sobre las aplicaciones del aprendizaje supervisado en la industria cinematográfica. 
   - Describe en un párrafo las variables más relevantes del dataset proporcionado y cómo podrían influir en las predicciones de ingresos y éxito.

---

### 2. **Regresión Lineal para la Predicción de Ingresos Brutos**
   #### 2.1 Selección de variables y análisis preliminar
   - Identifica las variables independientes más importantes que podrían influir en los ingresos brutos (variable dependiente). Justifica tu elección con análisis de correlación y gráficos de dispersión.

   #### 2.2 Entrenamiento y evaluación
   - Construye un modelo de regresión lineal usando las variables seleccionadas.
   - Evalúa el modelo con métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R²). Interpreta los coeficientes de regresión obtenidos.

   #### 2.3 Diagnóstico del modelo
   - Genera los gráficos de residuos vs. valores predichos y Q-Q plot.
   - Analiza si se cumplen los supuestos de linealidad, homoscedasticidad, independencia y normalidad de los errores.

---

### 3. **Regresión Logística para Clasificación de Películas**
   #### 3.1 Definición de éxito
   - Crea una nueva columna binaria que clasifique las películas como exitosas si sus ingresos están por encima del percentil 75, y no exitosas en caso contrario.

   #### 3.2 Construcción y evaluación del modelo
   - Construye un modelo de regresión logística para clasificar las películas.
   - Evalúa el modelo usando métricas como la precisión, sensibilidad, y F1 score. Genera y analiza la curva ROC y el valor AUC.

---

### 4. **Afinamiento y Evaluación Avanzada de Modelos**
   #### 4.1 Validación cruzada
   - Implementa validación cruzada para el modelo de regresión lineal y el modelo de regresión logística. Explica cómo mejora la generalización de los modelos.

   #### 4.2 Identificación de sobreajuste y subajuste
   - Analiza los resultados de los modelos para determinar si presentan sobreajuste o subajuste. Propón estrategias para abordar estos problemas.

   #### 4.3 Ajuste de hiperparámetros
   - Ajusta los hiperparámetros del modelo de regresión logística (como el umbral de clasificación) para mejorar su rendimiento.

---

### 5. **Extensión: K-Means para Segmentación de Películas**
   #### 5.1 Agrupamiento de películas
   - Utiliza el algoritmo de K-Means para agrupar películas en función de variables como presupuesto, likes en Facebook de actores principales y número de críticas.
   - Determina el número óptimo de clusters usando el método del codo y analiza los resultados.

   #### 5.2 Interpretación de los clusters
   - Describe las características principales de cada cluster identificado y cómo podrían usarse en estrategias de marketing cinematográfico.

---

### 6. **Conclusión**
   - Reflexiona sobre los resultados obtenidos con los modelos de regresión y K-Means.
   - Discute las limitaciones y posibles mejoras para futuros estudios.

---

## Instrucciones:
0. Actividad grupal: Mínimo 2, máximo 4 personas.
1. Documenta cada actividad en un cuaderno de Python o Jupyter Notebook.
2. Recuerda realizar la limpieza y preprocesamiento de datos necesarios para cada modelo.
3. Utiliza las bibliotecas de `scikit-learn`, `pandas`, `numpy`, `matplotlib` y `seaborn` para implementar los modelos y visualizaciones.
4. Usa gráficos y tablas para respaldar tus análisis y conclusiones.
5. Entrega tus resultados junto con el código desarrollado.
6. Asegúrate de que tu código sea claro y esté bien comentado.
7. **Fecha de entrega: 27 de noviembre de 2024, 23:59 hrs.**
8. **Formato de entrega: Crear un archivo ZIP con el cuaderno de Python y cualquier archivo adicional necesario (Dataset utilizado), guárdalo en un Google Drive que tenga acceso público, esto servirá como medio de evidencia para el programa Talento Tech. Asegúrate de tener un archivo con los nombres de los integrantes de tu equipo.**
9. Sube tu proyecto al repositorio de GitHub de tu equipo, copia y pega el enlace en el formulario que suministrará el profesor.

## Recursos Adicionales
- Dataset de películas: [Enlace al dataset](https://github.com/davidriveraarbelaez/IA_Explorador/blob/main/Datasets/movie_metadata.csv). 
- Implementaciones sugeridas: [Repositorio del curso](https://github.com/davidriveraarbelaez/IA_Explorador).

¡Buena suerte con la misión!