import pandas as pd
import numpy as np

np.random.seed(42)
data = []
id_cliente = 1

# Cliente momentáneo (0): muy poco gasto o muy poca frecuencia, pero agregamos ruido
for _ in range(100):
    gasto = np.random.normal(loc=50, scale=30)  # valores negativos posibles
    frecuencia = np.random.normal(loc=1, scale=1)
    data.append([id_cliente, max(gasto, 1), max(frecuencia, 0), 0])  # forzamos positivos
    id_cliente += 1

# Cliente regular (1): mezcla de gasto medio y frecuencia media, con ruido
for _ in range(100):
    gasto = np.random.normal(loc=150, scale=40)
    frecuencia = np.random.normal(loc=3, scale=1)
    data.append([id_cliente, max(gasto, 1), max(frecuencia, 0), 1])
    id_cliente += 1

# Cliente seguro (2): alto gasto y alta frecuencia, pero con más solapamiento con regular
for _ in range(100):
    gasto = np.random.normal(loc=300, scale=60)
    frecuencia = np.random.normal(loc=6, scale=1.5)
    data.append([id_cliente, max(gasto, 1), max(frecuencia, 0), 2])
    id_cliente += 1

# Guardar CSV
df = pd.DataFrame(data, columns=["id_cliente", "gasto_promedio", "frecuencia_semanal", "categoria"])
df.to_csv("clientes_dificiles.csv", index=False)
print("✅ Datos con complejidad generados.")
