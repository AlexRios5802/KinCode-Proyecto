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

# Extraer nombres
nombres = data['nombre'].values

# Separar variables y etiquetas
X = data[['gasto_promedio', 'frecuencia_semanal']].values
y = data['categoria'].values

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a tensores
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dividir datos
X_train, X_test, y_train, y_test, nombres_train, nombres_test = train_test_split(
    X_tensor, y_tensor, nombres, test_size=0.2, random_state=42
)

# Modelo
class ClienteNet(nn.Module):
    def __init__(self):
        super(ClienteNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializar modelo
model = ClienteNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluación
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Etiquetas
categorias = {
    0: "Cliente momentáneo",
    1: "Cliente regular",
    2: "Cliente seguro"
}

print("\nClasificación de clientes:")
for nombre, pred in zip(nombres_test, predicted):
    print(f"{nombre:<10} -> {categorias[int(pred)]}")
