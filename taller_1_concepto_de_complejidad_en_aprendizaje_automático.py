# -*- coding: utf-8 -*-
"""Taller 1 - Concepto de complejidad en aprendizaje automático.ipynb

DIEGO MENESES 

# Taller 1 - Concepto de complejidad en aprendizaje automático

## Etapa 1: Entendimiento de los datos
"""

# Carga de librerías y lectura del archivo que contiene los datos

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None, na_values=" ?")

# Se etiquetan las columnas para poder interpretar más fácilmente los datos.
# Cada fila es un ejemplo, cada columna un atributo.
# La columna llamada C es el atributo a predecir
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data.head()

#Cuál es el número de registros?
#Cuál es el número de atributos?

shape = data.shape
shape

#Cual es el tipo de los atributos?
data.dtypes

#Medida de centralidad y desviación para atributos numéricos:
data.describe()

# Diagrama de cajas y bigotes para atributos numéricos
# Permite identificar la existencia de datos atípicos

plt.boxplot((data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']))
plt.show()

#Medida de centralidad para atributos categóricos:
data.mode()

# Correlación entre los atributos de entrada numéricos
# Permite detectar si hay atributos redundantes (correlación mayor a 0.85 o menor a -0.85)
data.corr()

# Cual es el máximo de datos faltantes en un mismo registro?
# Si hay registros a los que les faltan muchos valores, es mejor eliminarlos.
max(data.isnull().sum(axis=1))

#Cuantos datos faltantes hay por cada atributo?
data.isnull().sum()

#Cuantos registros hay por cada clase? es decir, por cada valor del atributo de salida?

print(data['class'].value_counts())

"""## Etapa 2: Preparación de los datos





"""

from sklearn.preprocessing import LabelEncoder

# Se convierten los atributos categóricos a valores numéricos
labelencoder= LabelEncoder()
data['class'] = labelencoder.fit_transform(data['class'])
data.head()

from sklearn import preprocessing

# Se normalizan los tres atributos seleccionados
data['sepal_length'] = preprocessing.scale(data['sepal_length'])
data['sepal_width'] = preprocessing.scale(data['sepal_width'])
data['petal_length'] = preprocessing.scale(data['petal_length'])
data['petal_width'] = preprocessing.scale(data['petal_width'])
data.head()

# Se hace balanceo de clases eliminando ejemplos de la clase mayoritaria
g = data.groupby(data['class'])
dataBal = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
dataBal

# Se verifica que haya quedado el mismo número de registros por cada clase
print(dataBal['class'].value_counts())

"""## Etapa 3: Modelado

"""

# Dividir el conjunto de datos en conjuntos de entrenamiento y test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataBal.drop('class', axis=1), dataBal['class'], test_size=0.30)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Entrenamiento del modelo de clasificación por regresión logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logisticRegr = LogisticRegression(solver="lbfgs", max_iter=500, tol=0.01)
logisticRegr.fit(X_train, y_train)

# Aplicación del modelo construido a los datos de test
predictions = logisticRegr.predict(X_test)
predictions

# Cálculo del accuracy para evaluar el desempeño del modelo sobre los datos de test
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)

# Lista para almacenar los valores de precisión
accuracies = []

# Rango de valores de n_neighbors que queremos probar
neighbor_values = range(1, 26)

# Entrenar un modelo KNN con diferentes valores de n_neighbors
for n_neighbors in neighbor_values:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    print(f"Accuracy for n_neighbors={n_neighbors}: {accuracy}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(neighbor_values, accuracies, marker='o', linestyle='dashed', color='blue', markersize=10)
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

"""**Análisis:**
El modelo con 5 vecinos es el mejor, alcanzando una precisión del 97.8%. Este número de vecinos ofrece un valor adecuado para evitar el sobreajuste y tener una buena generalización.
"""

# Crear el clasificador KNN con 5 vecinos
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo con el conjunto de entrenamiento
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión usando Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labelencoder.classes_, yticklabels=labelencoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

"""**Métricas**

Recall (Sensibilidad): Indica la proporción de positivos reales que fueron identificados correctamente por el modelo. Es útil cuando las consecuencias de omitir positivos verdaderos son graves.

Precisión: Muestra la proporción de positivos identificados que eran realmente positivos. Es importante cuando el costo de un falso positivo es alto.

Exactitud (Accuracy): Mide la proporción de predicciones correctas (tanto positivas como negativas) en relación con el total de predicciones. Es una medida general del rendimiento del modelo.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calcula las métricas usando las predicciones y los valores reales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' para multiclase, considera el balance entre clases
recall = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

"""# Fin del programa"""