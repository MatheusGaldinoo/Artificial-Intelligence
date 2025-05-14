# Matheus Galdino de Souza - 14.05.2025

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Carragamento do database
irisds = load_iris()

X = irisds.data # Características que representam cada flor: Features
y = irisds.target # Classificação dessas flores: Rótulos

# Treinamento
model = GaussianNB() # Estratégia da Distribuição Normal
model.fit(X, y)

# Dados para realizar as predições
manual_data = [
    [5.8, 3.6, 1.5, 0.3],  # Provavelmente setosa
    [6.3, 2.7, 4.3, 1.2],  # Provavelmente versicolor
    [7.0, 3.2, 5.6, 2.0],  # Provavelmente virginica
    [5.5, 3.8, 1.9, 0.4],  # Provavelmente setosa
    [6.1, 2.9, 4.5, 1.1],  # Provavelmente versicolor
    [6.8, 3.0, 5.2, 2.3],  # Provavelmente virginica
]

predictions = model.predict(manual_data) 

for i in range(len(predictions)):
    print(f"Exemplo {i+1}: classe {predictions[i]} ({irisds.target_names[predictions[i]]})")
