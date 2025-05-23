from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Carregando o dataset

iris = datasets.load_iris()

# Separação dos dados

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

# Ciclos de treinamento, teste e avaliação de eficácia

results = []

for k in range(1, 30): # 0 vizinhos não é válido
  knn_model = KNeighborsClassifier(n_neighbors=k)
  knn_model.fit(X_train, y_train)

  y_pred = knn_model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  results.append(accuracy)

# Para plotar o gráfico...

plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), results, marker='o', linestyle='-', color='red')
plt.xlabel('k (número de vizinhos)')
plt.ylabel('Acurácia')
plt.xticks(range(1, 30))
plt.grid(True)
plt.show()

# Com base nos resultado deste experimento, um bom valor seria 12.
# Vale ressaltar que outros também apresentaram um bom resultado
