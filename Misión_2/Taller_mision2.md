**Taller: Predicción de exito de películas en el cine con Técnicas de Regresión**

Objetivo:
En este taller, exploraremos cómo construir modelos de regresión para predecir el rendimiento de una película en taquilla y su aceptación en IMDb. El 80% del tiempo estará dedicado a la regresión lineal, enfocándonos en la predicción de ingresos brutos, mientras que en el 20% restante aplicaremos regresión logística para clasificar películas como exitosas o no exitosas.

**Enunciados del Taller**

1. Introducción a la Regresión y al Dataset de Películas
   - Discusión breve sobre las aplicaciones de la regresión en la industria cinematográfica.
   - Explicación del dataset: qué representan las variables, su naturaleza y relevancia en modelos de predicción.

2. Regresión Lineal para la Predicción de Ingresos Brutos
   - Selección de Variable Objetivo y Variables Predictoras: Identificar la variable gross (ingresos brutos) como objetivo y seleccionar variables predictoras como budget (presupuesto), num_critic_for_reviews y actor_1_facebook_likes. Discutir la posible influencia de estas variables en los ingresos de una película.
   - Análisis de Correlación y Visualización de Relaciones: Crear gráficos de dispersión y analizar la correlación entre gross y otras variables, buscando identificar patrones que sugieran relaciones lineales.
   - Construcción del Modelo de Regresión Lineal: Entrenar un modelo de regresión lineal para predecir gross a partir de las variables seleccionadas. Discutir el proceso de ajuste de los coeficientes del modelo.
   - Evaluación e Interpretación del Modelo:
       - Explicar las métricas de evaluación como el error cuadrático medio (MSE) y el coeficiente de determinación (R²) para medir la precisión del modelo.
       - Analizar los coeficientes de las variables predictoras e interpretar su influencia en los ingresos brutos.
       - Aplicaciones Prácticas y Predicción: Realizar predicciones de ingresos brutos para algunas películas del dataset y discutir cómo esta información podría usarse en la toma de decisiones en la industria cinematográfica.

3. Regresión Logística para Clasificación de Películas (20% del Taller)
  - Definición de la Variable Binaria "Éxito de una Película": Crear una nueva variable binaria que clasifique las películas en "exitosas" (si gross supera un umbral, como el percentil 75) y "no exitosas" (si está por debajo de ese umbral).
  - Construcción y Evaluación del Modelo de Regresión Logística: Entrenar un modelo de regresión logística usando variables como budget, num_critic_for_reviews y actor_1_facebook_likes. Evaluar el modelo con métricas de clasificación (precisión, sensibilidad, y matriz de confusión) para medir su capacidad de predecir si una película será exitosa o no.

4.  Conclusión y Discusión Final
    Reflexionar sobre los resultados obtenidos con ambos modelos y las limitaciones de cada enfoque para hacer predicciones en la industria del cine.
