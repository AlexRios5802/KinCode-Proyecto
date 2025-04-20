from fastapi import FastAPI, HTTPException, Path #agregado
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import json
from typing import List #agregado
import datetime

app = FastAPI()

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Asegúrate de ajustar esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión a la base de datos
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="Orders",
        user="postgres",
        password="admin"
    )


@app.get("/pedidos") #pedidos-listos 
def obtener_pedidos_listos():
    conn = get_db_connection()
    cursor = conn.cursor()

    #para que sea un pedido con determinado estado, agregar un WHERE antes del ORDER BY
    query = """ 
    SELECT 
        p.ID_Pedido,
        p.Canal,
        p.Estado_Pedido,
        pr.Nombre AS Producto,
        pr.Categoria,
        pp.Personalizacion,
        SUM(pp.Cantidad) OVER (PARTITION BY p.ID_Pedido) AS Total_productos
    FROM 
        Pedidos p
    JOIN 
        Productos_Pedido pp ON p.ID_Pedido = pp.ID_Pedido
    JOIN 
        Productos pr ON pp.ID_Producto = pr.ID_Producto
    ORDER BY 
        Total_Productos ASC, p.Fecha ASC;
    """
    #Total_Productos ASC, p.Fecha ASC;
    #SUM(pp.Cantidad) OVER (PARTITION BY p.ID_Pedido) AS Total_productos, despues de p.Personalizacion;
    #p.ID_Pedido
    #se ordenan de menor a mayor
    
    cursor.execute(query)
    rows = cursor.fetchall()

    pedidos_dict = {}

    for row in rows:
        id_pedido, canal, estado, producto, categoria, personalizacion, total_productos = row #total_productos

        if id_pedido not in pedidos_dict:
            pedidos_dict[id_pedido] = {
                "numeroOrden": id_pedido,
                "canal": canal,
                "estado": estado,
                "principal": [], #[]
                "complementos": [],
                "postre": [],
                "especificaciones": []
            }

        if categoria == "Principal":
            pedidos_dict[id_pedido]["principal"].append(producto)
        elif categoria == "Complemento":
            pedidos_dict[id_pedido]["complementos"].append(producto)
        elif categoria == "Postre":
            pedidos_dict[id_pedido]["postre"].append(producto)
            
        #cargar especificaciones
        try:
            espec = json.loads(personalizacion)
            if espec:
                if isinstance(espec.get("detalle"), list):
                    texto = ', '.join(espec["detalle"])
                else:
                    texto = str(espec)
                    pedidos_dict[id_pedido]["especificaciones"].append(texto)
                #texto = ', '.join([f"{k}: {v}" for k, v in espec.items()])
                #pedidos_dict[id_pedido]["especificaciones"].append(texto)
        except Exception:
            pass

    cursor.close()
    conn.close()
    
    #pedidos_lista = list(pedidos_dict.values())
    return list(pedidos_dict.values())
    #return pedidos_final
    
    #ordenamiento dinamico
    """ahora = datetime.datetime.now()
    
    def prioridad(pedido):
        #calcular antiguedad del pedido
        minutos_espera = (ahora - pedido["fecha"]).total_seconds() / 60
        #si el pedido espera mas de 5 minutos, se le asigna una prioridad
        if minutos_espera > 5 and pedido["total_productos"] > 5:
            return(0, pedido["fecha"]) #mayor prioridad
        else:
            return(1, pedido["total_productos"]), pedido["fecha"] #ordenamiento normal, por cantidad y fecha
    #primero la prioridad de pedidos grandes
    pedidos_lista.sort(key=prioridad)
    
    #dentro de pedidos con misma prioridad, los pequeños van primero
    pedidos_lista.sort(key=lambda x: (x["total_productos"], x["fecha"]))
    
    return pedidos_lista"""

#clase para parsear los productos
class ProductoItem(BaseModel):
    nombre: str
    cantidad: int = 1
    especificaciones: List[str] = [] #lista de las especificaciones

#clase para crear pedidos
class PedidoCreate(BaseModel):
    numeroOrden: int
    principal: List[ProductoItem] #antes List[str]
    complementos: List[ProductoItem]
    postre: List[ProductoItem]
    canal: str
    especificaciones: List[str]
    estado: str   

@app.post("/pedidos/")
def crear_pedido(pedido: PedidoCreate):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Insertar en la tabla Pedidos
        cursor.execute(
            "INSERT INTO Pedidos (ID_Pedido, Canal, Estado_Pedido, Fecha) VALUES (%s, %s, %s, %s)",
            (pedido.numeroOrden, pedido.canal, pedido.estado, datetime.datetime.now())
        )

        # Insertar en Productos_Pedido
        for categoria, productos in [
            ("Principal", pedido.principal),
            ("Complemento", pedido.complementos),
            ("Postre", pedido.postre)
        ]:
            for item in productos: #nombre in productos
                nombre = item.nombre.strip()
                cantidad = item.cantidad
                
                # Buscar el ID del producto
                cursor.execute(
                    "SELECT ID_Producto FROM Productos WHERE Nombre = %s AND Categoria = %s",
                    (nombre, categoria) #nombre.strip()
                )
                resultado = cursor.fetchone()
                if not resultado:
                    continue  # Si el producto no existe, lo ignoramos
                id_producto = resultado[0]

                # Buscar especificaciones relacionadas
                personalizacion = {}
                if item.especificaciones:
                    personalizacion["detalle"] = item.especificaciones #se guarda como una lista directamente
                
                #for espec in pedido.especificaciones:
                    #if nombre in espec:
                        #partes = espec.split(":")
                        #if len(partes) == 2:
                            #personalizacion["detalle"] = partes[1].strip()

                #para ver que valores se estan mandando
                print("Valores para insertar:", (pedido.numeroOrden, id_producto, 1, "{}", categoria))
                
                # Insertar producto con posible personalización
                print(personalizacion) #verificar como se imprime la personalizacion
                cursor.execute(
                    #por defecto se agrega un pedido , si se quieren agregar más hay q cambiar el como se pide en el html
                    #el valor de personalizacion deberia ser json.dumps(personalizacion)
                    #por el momento se deja vacio, por practicidad
                    "INSERT INTO Productos_Pedido (ID_Pedido, ID_Producto, Cantidad, Personalizacion) VALUES (%s, %s, %s, %s)",
                    (pedido.numeroOrden, id_producto, cantidad, json.dumps(personalizacion)) #"{}"
                )

        conn.commit()
        return {"mensaje": "Pedido creado correctamente"}

    except Exception as e:
        conn.rollback()
        print("Error al guardar pedido:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cursor.close()
        conn.close()
        
class EstadoUpdate(BaseModel):
    estado: str        

#ruta para actualizar el estado del pedido
@app.put("/pedidos/{id_pedido}/estado") #int
def actualizar_estado_pedido(id_pedido: int, estado_update: EstadoUpdate): #str
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE Pedidos SET Estado_Pedido = %s WHERE ID_pedido = %s",
            (estado_update.estado, id_pedido) #nuevo_estado, id_pedido
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Pedido no encontrado")
        
        conn.commit()
        return {"mensaje": "Estado actualizado correctamente :)"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
