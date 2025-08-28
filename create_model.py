import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Definimos una clase personalizada de dataset heredando de torch.utils.data.Dataset
# Esto permite crear un dataset personalizado compatible con PyTorch para cargar datos de manera eficiente
class MyDataset(Dataset):
    def __init__(self):
        # Inicializamos los datos: una lista de diccionarios con 'name' (texto) y 'age' (entero)
        # Creamos 100 elementos con nombres "Item 0" a "Item 99" y edades de 0 a 99
        self.data = [{"name": f"Item {i}", "age": i} for i in range(100)]

    def __len__(self):
        # Método obligatorio que devuelve el tamaño total del dataset
        # En este caso, 100 elementos (longitud de self.data)
        return len(self.data)

    def __getitem__(self, idx):
        # Método obligatorio que devuelve un elemento del dataset en la posición 'idx'
        # Retorna un diccionario con la edad ('age') y una etiqueta binaria ('label')
        # La etiqueta es 1 si la edad es >= 18 (mayor de edad), 0 si es < 18 (menor de edad)
        item = self.data[idx]
        label = 1 if item['age'] >= 18 else 0
        return {'age': item['age'], 'label': label}

# Definimos una red neuronal simple para clasificación binaria
# Hereda de nn.Module, la clase base de PyTorch para modelos neuronales
class AgeClassifier(nn.Module):
    def __init__(self):
        # Llamamos al constructor de la clase base (nn.Module)
        super(AgeClassifier, self).__init__()
        # Definimos la primera capa lineal: toma 1 entrada (edad) y produce 10 salidas (neuronas)
        self.fc1 = nn.Linear(1, 10)
        # Definimos la activación ReLU, que introduce no linealidad (convierte valores negativos a 0)
        self.relu = nn.ReLU()
        # Definimos la segunda capa lineal: toma 10 entradas (de la capa anterior) y produce 1 salida
        self.fc2 = nn.Linear(10, 1)
        # Definimos la activación sigmoide, que mapea la salida a un rango [0,1] (probabilidad)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Método que define el flujo de datos a través de la red (paso hacia adelante)
        # x: tensor de entrada con forma [batch_size, 1] (edad)
        x = self.fc1(x)  # Aplicamos la primera capa lineal
        x = self.relu(x)  # Aplicamos ReLU
        x = self.fc2(x)  # Aplicamos la segunda capa lineal
        x = self.sigmoid(x)  # Aplicamos sigmoide para obtener una probabilidad
        return x  # Salida: tensor con forma [batch_size, 1]

# Creamos el dataset y el DataLoader
# Instanciamos el dataset personalizado
dataset = MyDataset()
# Creamos un DataLoader para manejar lotes de datos
# batch_size=32: procesa 32 ejemplos a la vez
# shuffle=True: mezcla los datos en cada época para mejorar el entrenamiento
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instanciamos el modelo, la función de pérdida y el optimizador
# Creamos una instancia del modelo AgeClassifier
model = AgeClassifier()
# Definimos la función de pérdida: Binary Cross Entropy (BCE) para clasificación binaria
criterion = nn.BCELoss()
# Definimos el optimizador: SGD (Stochastic Gradient Descent) con tasa de aprendizaje 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamos el modelo
num_epochs = 300  # Número de épocas (iteraciones completas sobre el dataset)
model.train()  # Ponemos el modelo en modo entrenamiento (activa capas como Dropout, si las hubiera)
for epoch in range(num_epochs):
    total_loss = 0  # Acumulador para la pérdida total en la época
    correct = 0  # Acumulador para predicciones correctas
    total = 0  # Acumulador para el número total de ejemplos
    for batch in dataloader:
        # Obtenemos los datos del lote
        # ages: tensor de edades con forma [batch_size], lo convertimos a [batch_size, 1]
        ages = batch['age'].float().unsqueeze(1)
        # labels: tensor de etiquetas con forma [batch_size], lo convertimos a [batch_size, 1]
        labels = batch['label'].float().unsqueeze(1)

        # Reiniciamos los gradientes acumulados en el optimizador
        optimizer.zero_grad()

        # Paso hacia adelante: obtenemos las predicciones del modelo
        outputs = model(ages)
        # Calculamos la pérdida comparando las predicciones con las etiquetas reales
        loss = criterion(outputs, labels)

        # Paso hacia atrás: calculamos los gradientes de la pérdida respecto a los parámetros
        loss.backward()
        # Actualizamos los pesos del modelo usando el optimizador
        optimizer.step()

        # Acumulamos la pérdida del lote
        total_loss += loss.item()

        # Calculamos la precisión del lote
        # Convertimos las probabilidades a predicciones binarias (1 si >= 0.5, 0 si < 0.5)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)  # Sumamos el número de ejemplos en el lote
        correct += (predicted == labels).sum().item()  # Sumamos las predicciones correctas

    # Cada 10 épocas, imprimimos la pérdida promedio y la precisión
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)  # Pérdida promedio por lote
        accuracy = 100 * correct / total  # Precisión en porcentaje
        print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")

# Evaluamos el modelo en todo el dataset
model.eval()  # Ponemos el modelo en modo evaluación (desactiva Dropout, si lo hubiera)
correct = 0  # Reiniciamos el contador de predicciones correctas
total = 0  # Reiniciamos el contador de ejemplos
with torch.no_grad():  # Desactivamos el cálculo de gradientes para ahorrar memoria y tiempo
    for batch in dataloader:
        # Obtenemos los datos del lote
        ages = batch['age'].float().unsqueeze(1)
        labels = batch['label'].float().unsqueeze(1)
        # Realizamos la predicción
        outputs = model(ages)
        # Convertimos las probabilidades a predicciones binarias
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)  # Sumamos los ejemplos del lote
        correct += (predicted == labels).sum().item()  # Sumamos las predicciones correctas

# Calculamos e imprimimos la precisión final
accuracy = 100 * correct / total
print(f"\nPrecisión final en el dataset: {accuracy:.2f}%")

# Guardamos los pesos del modelo en un archivo
# Esto guarda solo los parámetros (pesos y sesgos) del modelo, no la arquitectura
torch.save(model.state_dict(), 'age_classifier_weights.pth')
print("Modelo guardado como 'age_classifier_weights.pth'")

# Comentario sobre cómo cargar el modelo más tarde
# Para cargar los pesos, se debe crear una nueva instancia de AgeClassifier
# y usar load_state_dict para cargar los pesos guardados
# new_model = AgeClassifier()
# new_model.load_state_dict(torch.load('age_classifier_weights.pth'))
# new_model.eval()