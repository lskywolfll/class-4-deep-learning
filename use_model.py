import torch
import torch.nn as nn


# Define the AgeClassifier model (must match the original architecture)
class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input: 1 feature (age), Output: 10 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)  # Output: 1 neuron for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Function to load the model
def load_model(model_path):
    model = AgeClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Function to predict if the person is of legal age
def predict_age_classification(model, age):
    # Convert age to tensor and add batch dimension
    age_tensor = torch.tensor([[float(age)]], dtype=torch.float32)
    with torch.no_grad():
        output = model(age_tensor)
        prediction = (output >= 0.5).float().item()
        return "mayor de edad" if prediction == 1.0 else "menor de edad"

# Main function to get user input and make prediction
def main():
    # Load the saved model
    model_path = 'age_classifier_weights.pth'
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'age_classifier_weights.pth'. Asegúrate de que el modelo esté guardado.")
        return

    # Get user input
    name = input("Ingrese su nombre: ")
    try:
        age = float(input("Ingrese su edad: "))
        if age < 0:
            print("Error: La edad no puede ser negativa.")
            return
    except ValueError:
        print("Error: Por favor, ingrese una edad válida (número).")
        return

    # Make prediction
    result = predict_age_classification(model, age)
    print(f"{name}, según el modelo, eres {result}.")

if __name__ == "__main__":
    main()