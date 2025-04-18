import pandas as pd
import numpy as np

np.random.seed(42)

data = []

# Cliente momentáneo
for _ in range(100):
    gasto = np.random.uniform(10, 100)
    frecuencia = np.random.randint(0, 2)
    data.append([gasto, frecuencia, 0])

# Cliente regular
for _ in range(100):
    gasto = np.random.uniform(100, 200)
    frecuencia = np.random.randint(2, 4)
    data.append([gasto, frecuencia, 1])

# Cliente seguro
for _ in range(100):
    gasto = np.random.uniform(200, 500)
    frecuencia = np.random.randint(4, 8)
    data.append([gasto, frecuencia, 2])

df = pd.DataFrame(data, columns=['gasto_promedio', 'frecuencia_semanal', 'categoria'])
df.to_csv("clientes_comida_rapida.csv", index=False)
print("Archivo creado exitosamente.")
