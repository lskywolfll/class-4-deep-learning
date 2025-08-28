
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, input):
        # Define the forward pass
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = Model()
data = torch.randn(1, 100)
result = model(data)
