import pandas as pd
import numpy as np
import random

np.random.seed(42)

nombres = ["Carlos", "Ana", "Luis", "Marta", "Pedro", "Laura", "Diego", "Sofía", "Andrés", "Lucía",
           "Javier", "Elena", "Manuel", "Valeria", "Miguel", "Camila", "David", "Paula", "Sebastián", "Natalia"]

data = []

# Cliente momentáneo
for _ in range(100):
    nombre = random.choice(nombres)
    gasto = np.random.uniform(10, 100)
    frecuencia = np.random.randint(0, 2)
    data.append([nombre, gasto, frecuencia, 0])

# Cliente regular
for _ in range(100):
    nombre = random.choice(nombres)
    gasto = np.random.uniform(100, 200)
    frecuencia = np.random.randint(2, 4)
    data.append([nombre, gasto, frecuencia, 1])

# Cliente seguro
for _ in range(100):
    nombre = random.choice(nombres)
    gasto = np.random.uniform(200, 500)
    frecuencia = np.random.randint(4, 8)
    data.append([nombre, gasto, frecuencia, 2])

df = pd.DataFrame(data, columns=['nombre', 'gasto_promedio', 'frecuencia_semanal', 'categoria'])
df.to_csv("clientes_comida_rapida.csv", index=False)
print("Archivo con nombres generado.")
