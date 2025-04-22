import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd

# Leer datos difíciles
data = pd.read_csv("clientes_dificiles.csv")

ids = data["id_cliente"].values
X = data[["gasto_promedio", "frecuencia_semanal"]].values
y = data["categoria"].values

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# División
X_train, X_test, y_train, y_test, _, _ = train_test_split(
    X_tensor, y_tensor, ids, test_size=0.2, random_state=42
)

# Modelo
class ClienteNet(nn.Module):
    def __init__(self):
        super(ClienteNet, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

model = ClienteNet()
print("Estructura del modelo:\n", model)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validación
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    if epoch % 10 == 0:
        print(f"Época {epoch}, Pérdida entrenamiento: {loss.item():.4f}, Test: {test_loss.item():.4f}")

    # Early stopping
    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping en época {epoch}")
            break




# Guardar el modelo
torch.save(model.state_dict(), "modelo_clientes.pth")
print("Modelo guardado como 'modelo_clientes.pth'")

# Evaluación
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)

print("\nReporte de clasificación:\n")
print(classification_report(y_test, predictions))
