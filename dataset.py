import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Definimos una clase personalizada de dataset heredando de torch.utils.data.Dataset
class MyDataset(Dataset):
    def __init__(self):
        # Inicializamos los datos: lista de diccionarios con 'name' y 'age'
        self.data = [{"name": f"Item {i}", "age": i} for i in range(100)]

    def __len__(self):
        # Devuelve la cantidad de elementos en el dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Obtenemos el elemento en la posición 'idx'
        item = self.data[idx]
        # Creamos una etiqueta binaria: 1 si age >= 18, 0 si age < 18
        label = 1 if item['age'] >= 18 else 0
        return {'age': item['age'], 'label': label}

# Definimos una red neuronal simple para clasificación binaria
class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        # Una capa lineal que toma 'age' como entrada (1 característica) y produce 1 salida
        self.fc1 = nn.Linear(1, 10)  # Capa oculta con 10 neuronas
        self.relu = nn.ReLU()        # Activación ReLU
        self.fc2 = nn.Linear(10, 1)  # Capa de salida (1 neurona para clasificación binaria)
        self.sigmoid = nn.Sigmoid()  # Activación sigmoide para obtener probabilidad [0,1]

    def forward(self, x):
        # Definimos el paso hacia adelante
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Creamos el dataset y el DataLoader
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instanciamos el modelo, la función de pérdida y el optimizador
model = AgeClassifier()
criterion = nn.BCELoss()  # Pérdida de entropía cruzada binaria
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizador SGD con tasa de aprendizaje 0.01

# Entrenamos el modelo
num_epochs = 50  # Número de épocas
model.train()  # Ponemos el modelo en modo entrenamiento
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        # Obtenemos las edades y etiquetas del lote
        ages = batch['age'].float().unsqueeze(1)  # Convertimos a tensor y añadimos dimensión [batch_size, 1]
        labels = batch['label'].float().unsqueeze(1)  # Convertimos etiquetas a tensor [batch_size, 1]

        # Reiniciamos los gradientes
        optimizer.zero_grad()

        # Paso hacia adelante: predicciones del modelo
        outputs = model(ages)
        loss = criterion(outputs, labels)

        # Paso hacia atrás: calculamos gradientes y actualizamos pesos
        loss.backward()
        optimizer.step()

        # Calculamos la pérdida total
        total_loss += loss.item()

        # Calculamos la precisión
        predicted = (outputs >= 0.5).float()  # Predicción: 1 si salida >= 0.5, 0 si no
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Imprimimos la pérdida y precisión por época
    if (epoch + 1) % 10 == 0:  # Mostramos cada 10 épocas
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")

# Evaluamos el modelo en todo el dataset
model.eval()  # Ponemos el modelo en modo evaluación
correct = 0
total = 0
with torch.no_grad():  # Desactivamos el cálculo de gradientes para evaluación
    for batch in dataloader:
        ages = batch['age'].float().unsqueeze(1)
        labels = batch['label'].float().unsqueeze(1)
        outputs = model(ages)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nPrecisión final en el dataset: {accuracy:.2f}%")