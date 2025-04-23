import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

# Configuración inicial
np.random.seed(42)
torch.manual_seed(42)
NUM_CLIENTES = 1000
NUM_PRODUCTOS = 50
MAX_PEDIDOS_POR_PRODUCTO = 20
NUM_REGISTROS = 50000

## Generador de datos sintéticos (versión actualizada)
def generar_datos_sinteticos(num_clientes, num_productos, num_registros, max_pedidos):
    # Generar IDs de clientes y productos
    clientes = [f'CLI_{i:04d}' for i in range(1, num_clientes+1)]
    productos = [f'PROD_{i:03d}' for i in range(1, num_productos+1)]
    
    # Lista para almacenar registros
    registros = []
    
    # Generar registros aleatorios
    for _ in range(num_registros):
        cliente = np.random.choice(clientes)
        producto = np.random.choice(productos)
        
        # Asignar más pedidos a algunos productos (para crear patrones)
        if producto in productos[:10]:  # Los primeros 10 productos son más populares
            num_pedidos = np.random.randint(5, max_pedidos+1)
        else:
            num_pedidos = np.random.randint(1, max_pedidos//2 + 1)
            
        # Añadir preferencias personales basadas en el ID del cliente
        cliente_num = int(cliente.split('_')[1])
        if cliente_num % 7 == 0 and producto in productos[10:20]:
            num_pedidos += np.random.randint(3, 8)
        elif cliente_num % 5 == 0 and producto in productos[20:30]:
            num_pedidos += np.random.randint(2, 6)
            
        registros.append({
            'cliente_id': cliente,
            'producto_id': producto,
            'num_pedidos': num_pedidos
        })
    
    # Crear DataFrame y agrupar
    datos = pd.DataFrame(registros)
    datos = datos.groupby(['cliente_id', 'producto_id'], as_index=False).sum()
    
    return datos, clientes, productos

# Generar los datos
datos, clientes, productos = generar_datos_sinteticos(
    NUM_CLIENTES, NUM_PRODUCTOS, NUM_REGISTROS, MAX_PEDIDOS_POR_PRODUCTO
)

print(f"Datos generados: {len(datos)} registros únicos")
print(datos.head())

## Preprocesamiento de datos
# Codificar clientes y productos
cliente_encoder = LabelEncoder()
producto_encoder = LabelEncoder()

datos['cliente_encoded'] = cliente_encoder.fit_transform(datos['cliente_id'])
datos['producto_encoded'] = producto_encoder.fit_transform(datos['producto_id'])

# Normalizar el número de pedidos
scaler = MinMaxScaler()
datos['num_pedidos_norm'] = scaler.fit_transform(datos[['num_pedidos']])

# Dividir en entrenamiento y prueba
train, test = train_test_split(datos, test_size=0.2, random_state=42)

# Dataset personalizado para PyTorch
class ClienteProductoDataset(Dataset):
    def __init__(self, df):
        self.clientes = torch.LongTensor(df['cliente_encoded'].values)
        self.productos = torch.LongTensor(df['producto_encoded'].values)
        self.pedidos = torch.FloatTensor(df['num_pedidos_norm'].values)
        
    def __len__(self):
        return len(self.clientes)
    
    def __getitem__(self, idx):
        return {
            'cliente': self.clientes[idx],
            'producto': self.productos[idx],
            'pedidos': self.pedidos[idx]
        }

# Crear datasets y dataloaders
train_dataset = ClienteProductoDataset(train)
test_dataset = ClienteProductoDataset(test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

## Modelo en PyTorch
class RecomendacionModel(nn.Module):
    def __init__(self, num_clientes, num_productos, embedding_size=32):
        super(RecomendacionModel, self).__init__()
        self.embedding_cliente = nn.Embedding(num_clientes, embedding_size)
        self.embedding_producto = nn.Embedding(num_productos, embedding_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_size*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regularización L2
        self.reg = nn.ModuleList()
        self.reg.append(self.embedding_cliente)
        self.reg.append(self.embedding_producto)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                self.reg.append(layer)
        
    def forward(self, cliente, producto):
        cliente_embed = self.embedding_cliente(cliente)
        producto_embed = self.embedding_producto(producto)
        
        x = torch.cat([cliente_embed, producto_embed], dim=1)
        x = self.fc_layers(x)
        return x.squeeze()
    
    def l2_regularization(self, lambda_reg=0.001):
        l2_loss = 0
        for layer in self.reg:
            l2_loss += torch.norm(layer.weight, p=2)
        return lambda_reg * l2_loss

# Construir el modelo
modelo = RecomendacionModel(len(clientes), len(productos))
print(modelo)

## Entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = modelo.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, epochs=50, patience=5):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            cliente = batch['cliente'].to(device)
            producto = batch['producto'].to(device)
            target = batch['pedidos'].to(device)
            
            optimizer.zero_grad()
            output = model(cliente, producto)
            loss = criterion(output, target)
            loss += model.l2_regularization()  # Añadir regularización L2
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * cliente.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validación
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                cliente = batch['cliente'].to(device)
                producto = batch['producto'].to(device)
                target = batch['pedidos'].to(device)
                
                output = model(cliente, producto)
                loss = criterion(output, target)
                test_loss += loss.item() * cliente.size(0)
        
        test_loss /= len(test_loader.dataset)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Cargar el mejor modelo
    model.load_state_dict(torch.load('best_model.pth'))
    return model

modelo = train_model(modelo, train_loader, test_loader)

## Sistema de recomendación
class SistemaRecomendacion:
    def __init__(self, modelo, cliente_encoder, producto_encoder, scaler, datos_original):
        self.modelo = modelo
        self.cliente_encoder = cliente_encoder
        self.producto_encoder = producto_encoder
        self.scaler = scaler
        self.datos_original = datos_original
        self.productos_disponibles = producto_encoder.classes_
        self.modelo.eval()  # Poner el modelo en modo evaluación
        
    def recomendar(self, cliente_id, top_n=5):
        try:
            # Verificar si el cliente existe
            if cliente_id not in self.cliente_encoder.classes_:
                return f"Cliente {cliente_id} no encontrado. No se pueden hacer recomendaciones."
            
            # Codificar el cliente
            cliente_encoded = self.cliente_encoder.transform([cliente_id])[0]
            
            # Crear tensores para todos los productos
            cliente_tensor = torch.LongTensor([cliente_encoded] * len(self.productos_disponibles))
            producto_tensor = torch.arange(len(self.productos_disponibles), dtype=torch.long)
            
            # Predecir preferencias
            with torch.no_grad():
                predicciones = self.modelo(cliente_tensor, producto_tensor).numpy()
            
            # Escalar inversamente para obtener el número estimado de pedidos
            predicciones_esc = self.scaler.inverse_transform(predicciones.reshape(-1, 1)).flatten()
            
            # Crear dataframe con resultados
            resultados = pd.DataFrame({
                'producto_id': self.productos_disponibles,
                'prediccion_pedidos': predicciones_esc
            })
            
            # Ordenar por predicción descendente
            resultados = resultados.sort_values('prediccion_pedidos', ascending=False)
            
            # Filtrar productos que el cliente ya ha pedido mucho
            historico_cliente = self.datos_original[self.datos_original['cliente_id'] == cliente_id]
            if not historico_cliente.empty:
                resultados = resultados.merge(
                    historico_cliente[['producto_id', 'num_pedidos']], 
                    on='producto_id', 
                    how='left'
                )
                resultados['num_pedidos'] = resultados['num_pedidos'].fillna(0)
                # Dar preferencia a productos no pedidos o poco pedidos
                resultados['score'] = resultados['prediccion_pedidos'] / (resultados['num_pedidos'] + 1)
                resultados = resultados.sort_values('score', ascending=False)
            
            # Tomar los top_n productos
            recomendaciones = resultados.head(top_n)
            
            return recomendaciones[['producto_id', 'prediccion_pedidos']]
        
        except Exception as e:
            return f"Error al generar recomendaciones: {str(e)}"

# Crear el sistema de recomendación
sistema_recomendacion = SistemaRecomendacion(
    modelo, cliente_encoder, producto_encoder, scaler, datos
)

## Guardar el modelo entrenado y componentes
def guardar_modelo(modelo, cliente_encoder, producto_encoder, scaler, filename='modelo_recomendacion.pth'):
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'cliente_encoder': cliente_encoder,
        'producto_encoder': producto_encoder,
        'scaler': scaler,
        'datos_original': datos,
        'model_class': RecomendacionModel,
        'num_clientes': len(clientes),
        'num_productos': len(productos)
    }, filename)
    print(f"Modelo guardado en {filename}")

guardar_modelo(modelo, cliente_encoder, producto_encoder, scaler)

## Cargar el modelo (ejemplo de cómo se haría)
def cargar_modelo(filename='modelo_recomendacion.pth'):
    checkpoint = torch.load(filename)
    modelo = checkpoint['model_class'](
        checkpoint['num_clientes'],
        checkpoint['num_productos']
    )
    modelo.load_state_dict(checkpoint['model_state_dict'])
    
    sistema = SistemaRecomendacion(
        modelo,
        checkpoint['cliente_encoder'],
        checkpoint['producto_encoder'],
        checkpoint['scaler'],
        checkpoint['datos_original']
    )
    return sistema

## Ejemplo de uso
print("\nEjemplo de recomendaciones:")
cliente_ejemplo = np.random.choice(clientes)
print(f"Recomendaciones para {cliente_ejemplo}:")
recomendaciones = sistema_recomendacion.recomendar(cliente_ejemplo)
print(recomendaciones)