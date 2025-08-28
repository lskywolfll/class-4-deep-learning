import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Definimos una clase personalizada de dataset heredando de torch.utils.data.Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = [{"name": f"Item {i}", "age": i} for i in range(100)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = 1 if item['age'] >= 18 else 0
        return {'age': item['age'], 'label': label}

# Definimos una red neuronal simple para clasificación binaria
class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamos el modelo
num_epochs = 105
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        ages = batch['age'].float().unsqueeze(1)
        labels = batch['label'].float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(ages)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")

# Evaluamos el modelo
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        ages = batch['age'].float().unsqueeze(1)
        labels = batch['label'].float().unsqueeze(1)
        outputs = model(ages)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nPrecisión final en el dataset: {accuracy:.2f}%")

# Guardamos los pesos del modelo
torch.save(model.state_dict(), 'age_classifier_weights.pth')
print("Modelo guardado como 'age_classifier_weights.pth'")

# Para cargar el modelo más tarde:
# new_model = AgeClassifier()
# new_model.load_state_dict(torch.load('age_classifier_weights.pth'))
# new_model.eval()