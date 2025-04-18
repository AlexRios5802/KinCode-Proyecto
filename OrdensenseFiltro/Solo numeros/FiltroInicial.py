import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Cargar los datos
data = pd.read_csv("clientes_comida_rapida.csv")

# Separar variables y etiquetas
X = data[['gasto_promedio', 'frecuencia_semanal']].values
y = data['categoria'].values  # 0: momentáneo, 1: regular, 2: seguro

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a tensores de PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Modelo de red neuronal simple
class ClienteNet(nn.Module):
    def __init__(self):
        super(ClienteNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)  # 3 categorías

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializar modelo, pérdida y optimizador
model = ClienteNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenar la red
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Época {epoch+1}, Pérdida: {loss.item():.4f}")

# Evaluar
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    print("\nClasificación:")
    print(classification_report(y_test, predicted))
