let pedidos = [];

function cargarPedidos() {
  const tbody = document.getElementById("orderBody");
  const noOrders = document.getElementById("noOrders");
  const orderContainer = document.getElementById("orderContainer");

  if (pedidos.length === 0) {
    noOrders.style.display = "block";
    orderContainer.style.display = "none";
    return;
  }

  noOrders.style.display = "none";
  orderContainer.style.display = "block";
  tbody.innerHTML = '';

  pedidos.forEach(pedido => {
    const row = document.createElement("tr");
    row.setAttribute('data-orden', pedido.numeroOrden); //aqui hubo camnbio

    //funcion para crear los botones de los productos
    const btnProductos = (productos, numeroOrden) => {
      if (!productos || productos.length === 0) return "N/A";
      
      return productos.map((producto) => {
        let nombreMostrar;
    
        if (typeof producto === 'string') {
          // Parsear si viene como string "2x Big Mac"
          const match = producto.match(/^(\d+)x\s*(.+)$/i);
          if (match) {
            nombreMostrar = `${match[1]}x ${match[2]}`;
          } else {
            nombreMostrar = producto;
          }
        } else if (producto && producto.nombre && producto.cantidad) {
          // Si viene como objeto {nombre, cantidad}
          nombreMostrar = `${producto.cantidad}x ${producto.nombre}`;
        } else {
          nombreMostrar = "Producto inválido";
        }
    
        return `<button class="btn-product ${getButtonColor(pedido.estado)}"
          data-estado="normal" data-orden="${numeroOrden}"
          data-producto="${nombreMostrar}"
          onclick="estadoBtn(this)">${nombreMostrar}</button>`;
      }).join("");
    };

    //actualizamos el selector y la columna del estado con el estado actual del pedido
    const estadoSelect = `
    <select class="select-opt" onchange="cambiarEstado(this.value, '${pedido.numeroOrden}')">
      <option value="Pendiente" ${pedido.estado === 'Pendiente' ? 'selected' : ''}>Pendiente</option>
      <option value="Procesando" ${pedido.estado === 'Procesando' ? 'selected' : ''}>Procesando</option>
      <option value="Listo" ${pedido.estado === 'Listo' ? 'selected' : ''}>Listo</option>
    </select>
    `;

    row.innerHTML = `
      <td>${pedido.numeroOrden}</td>
      <td>${pedido.estado}</td>
      <td class="product-cell">${btnProductos(pedido.principal, pedido.numeroOrden)}</td>
      <td class="product-cell">${btnProductos(pedido.complementos, pedido.numeroOrden)}</td>
      <td class="product-cell">${btnProductos(pedido.postre, pedido.numeroOrden)}</td>
      <td>${pedido.canal}</td>
      <td>${pedido.especificaciones.join("<br>")}</td>
      <td>
        <select class="select-opt" onchange="cambiarEstado(this.value, '${pedido.numeroOrden}')">
          <option value="Pendiente" ${pedido.estado === 'Pendiente' ? 'selected' : ''}>Pendiente</option>
          <option value="Procesando" ${pedido.estado === 'Procesando' ? 'selected' : ''}>Procesando</option>
          <option value="Listo" ${pedido.estado === 'Listo' ? 'selected' : ''}>Listo</option>
        </select>
      </td>
    `;

    tbody.appendChild(row);
  });
}

//CAMBIO ACA
//funcion para actualizar la lista de pedidos cada 30 segundos
setInterval(() => {
  fetch("http://localhost:8000/pedidos")  // Traemos los pedidos desde la API
    .then(response => response.json())
    .then(data => {
      // Revisamos si algún pedido grande lleva más de 5 minutos en espera
      const now = Date.now();
      pedidos = data.map(pedido => {
        const pedidoTiempo = new Date(pedido.fecha).getTime();
        const tiempoEspera = now - pedidoTiempo;

        if (tiempoEspera > 300000 && pedido.estado !== "Listo") {
          // Si el pedido tiene más de 5 minutos de espera, movemos al principio
          return { ...pedido, estado: "Pendiente" }; // Aquí actualizas el estado si lo deseas
        }
        return pedido;
      });

      // Ordenamos los pedidos
      pedidos.sort((a, b) => {
        if (a.principal.length === b.principal.length) {
          return new Date(a.fecha) - new Date(b.fecha); // Si tienen la misma cantidad de productos, ordenamos por fecha
        }
        return a.principal.length - b.principal.length; // Ordenamos por cantidad de productos
      });

      cargarPedidos();  // Actualizamos la tabla
    })
    .catch(error => {
      console.error("Error al cargar pedidos desde la API:", error);
    });
}, 30000);  // 30 segundos


//funcion para obtener el color del boton
function getButtonColor(estado){
  switch(estado){
    case 'Pendiente':
      return 'btn-pendiente';
    case 'Procesando':
      return 'btn-procesando';
    case 'Listo':
      return 'btn-listo';
    default:
      return '';
  }
}

function cambiarEstado(nuevoEstado, numeroOrden) {
  numeroOrden = parseInt(numeroOrden); //eliminar aca
  const pedido = pedidos.find(p => p.numeroOrden === numeroOrden);
    //pedido.estado = nuevoEstado;
    //cargarPedidos(); // Recarga la tabla para mostrar el nuevo estado
    if(pedido){
      pedido.estado = nuevoEstado;

      //FUNCION PARA CONFIRMAR PEDIDOS Y MANDARLOS AL FINAL DEL FORMULARIO
      //si el estado es "Listo", se muestra la alerta
     /* if(nuevoEstado === "Listo"){
        const confirmar = confirm("¿Esta seguro de que el pedido esta completo?");

        if(confirmar){
          //movemos el pedido al final de la lista
          pedidos = pedidos.filter(p => p.numeroOrden !== numeroOrden);
          pedidos.push(pedido);
          cargarPedidos(); //reseteamos la tabla
          this.reset();
        }else {
          //si el usuario no confirma, se regresa al estado de "Procesando"
          pedido.estado = 'Pendiente';
          cargarPedidos();
          this.reset();
        }
      }else{
        cargarPedidos();
      }*/

      //actualizar el color de botones en la fila
      const fila = document.querySelector(`tr[data-orden = "${numeroOrden}"]`);
      if(fila) {
        const botones = fila.querySelectorAll('.btn-product');
        botones.forEach(boton => {
          boton.classList.remove('btn-pendiente', 'btn-procesando', 'btn-listo');
          switch(nuevoEstado){
            case 'Pendiente':
              boton.classList.add('btn-pendiente');
              boton.setAttribute("data-estado", "normal"); //quitar si no funciona
              break;
            case 'Procesando':
              boton.classList.add('btn-procesando');
              boton.setAttribute("data-estado", "amarillo");
              break;
            case 'Listo':
              boton.classList.add('btn-listo');
              boton.setAttribute("data-estado", "verde");
              break;
          }
        });
        const estadoCell = fila.querySelector('td:nth-child(2)');
        if(estadoCell){
          estadoCell.textContent = nuevoEstado;
        }
      }
      //llamar al metodo del back
      fetch(`http://localhost:8000/pedidos/${numeroOrden}/estado`,{
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({estado: nuevoEstado})
      })
      .then(response => {
        if(!response.ok) throw new Error("Error al actualizar el estado en la db");
        return response.json();
      })
      .then(data => {
        console.log("Estado actualizado:", data);
        //actualizar localmente el pedido
        let pedido = pedidos.find(p => p.numero_orden == numeroOrden);
        if(pedido){
          pedido.estado = nuevoEstado
          cargarPedidos();
        }
      })
      .catch(error => {
        console.log("Error al actualizar el estado", error);
      });

      //cargarPedidos();
    }
  }

  function estadoBtn(boton) {
    const estados = {
      normal: { class: "", next: "amarillo" },
      amarillo: { class: "btn-amarillo", next: "verde" },
      verde: { class: "btn-verde", next: "verde" }
    };
  
    const estadoActual = boton.getAttribute("data-estado");
    const siguienteEstado = estados[estadoActual].next;
  
    // Resetear clases de color
    boton.classList.remove("btn-amarillo", "btn-verde", "btn-pendiente", "btn-procesando", "btn-listo");
  
    //CAMBIO ACA
    //boton.classList.add("btn-bold")

    if (siguienteEstado !== "normal") {
      boton.classList.add(estados[siguienteEstado].class);
    }
  
    boton.setAttribute("data-estado", siguienteEstado);
  
    // Obtener fila y botones
    const fila = boton.closest('tr');
    const botones = fila.querySelectorAll('.btn-product');
  
    const colores = Array.from(botones).map(b => b.getAttribute("data-estado"));
    const todosIguales = colores.every(c => c === siguienteEstado);
  
    if (todosIguales && siguienteEstado !== "normal") {
      // Determinar nuevo estado
      let nuevoEstado = "";
      if (siguienteEstado === "amarillo") nuevoEstado = "Procesando";
      if (siguienteEstado === "verde") nuevoEstado = "Listo";
  
      const numeroOrden = fila.getAttribute("data-orden");
  
      // Actualizar el estado en el front (celda y selector)
      const estadoCell = fila.querySelector('td:nth-child(2)');
      if (estadoCell) {
        estadoCell.textContent = nuevoEstado;
      }
  
      const selector = fila.querySelector("select");
      if (selector) {
        selector.value = nuevoEstado;
      }

        //actualizamos el objeto en memoria
        pedidos = pedidos.map(p =>
          p.numeroOrden == numeroOrden ? {...p, estado: nuevoEstado} : p
        );
  
        // Llamar al backend para guardar el nuevo estado
        fetch(`http://localhost:8000/pedidos/${numeroOrden}/estado`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ estado: nuevoEstado })
        })
        .then(response => {
          if (!response.ok) throw new Error("Error al actualizar el estado en el backend");
          return response.json();
        })
        .then(data => {
          console.log("Estado actualizado automáticamente:", data);
        })
        .catch(error => {
          console.error("Error al actualizar automáticamente el estado:", error);
        });
      }
    }


//

function simularLlegadaPedido() {
  setTimeout(() => {
    pedidos = [
      {
        numeroOrden: "000932284AM",
        principal: ["2x BigMac"],
        complementos: ["5x Papas Grandes"],
        postre: ["10x Cono de Helado (Vainilla)"],
        canal: "Mostrador",
        especificaciones: ["1x BigMac: sin cebolla, extra cebolla"],
        estado: "Pendiente"
      },
      {
        numeroOrden: "1002",
        principal: ["McPollo"],
        complementos: ["Papas chicas"],
        postre: [],
        canal: "Mostrador",
        especificaciones: ["Sin mayonesa"],
        estado: "Pendiente"
      }
    ];
    cargarPedidos();
  }, 3000);
}

window.onload =window.onload = () => {
  fetch("http://localhost:8000/pedidos") //pedidos-listos
    .then(response => response.json())
    .then(data => {
      pedidos = data.map(p => ({
        numeroOrden: p.numero_orden,
        principal: p.principal || [],
        complementos: p.complementos || [],
        postre: p.postre || [],
        canal: p.canal,
        especificaciones: p.especificaciones || [],
        estado: p.estado
      }));
      //pedidos = data; // Asigna los pedidos reales
      cargarPedidos(); // Los muestra en la tabla
    })
    .catch(error => {
      console.error("Error al cargar pedidos desde la API:", error);
    });
};   //simularLlegadaPedido;

/*funcion para parsear el input de especificaciones
function obtenerEspecificaciones(){
  const especificacionesInput = document.getElementById("especificaciones").value;
  const map = {};

  especificacionesInput.split(",").forEach(entry => {
    const [producto, detalle] = entry.split(":").map(x => x.trim());
    if(producto && detalle){
      map[producto.toLowerCase()] = detalle;
    }
  });

  return map;

}*/


//FUNCION para parsear los productos y las cantidades
function parsearProductos(producto, especificacionesG){
  return producto.split(",").map(item => {
    item = item.trim();
    const match = item.match(/^(\d+)x\s*(.+)$/i);
    const nombre = match ? match[2].trim() : item;
    const cantidad = match ? parseInt(match[1]) : 1;

    return{
      nombre: nombre,
      cantidad: cantidad,
      especificaciones: especificacionesG
    };
    //if(match){
      //return { nombre: match[2].trim(), cantidad: parseInt(match[1]) };
    //} else{
      //return { nombre: item, cantidad: 1};
  }).filter(p => p.nombre);
}


document.getElementById("formPedido").addEventListener("submit", function (e) {
    e.preventDefault(); // Evita que se recargue la página
  
    //CAMBIO ACA
    const especificacionesInput = document.getElementById("especificaciones").value;
    const especificacionesG = especificacionesInput.split(",").map(item => item.trim()).filter(Boolean);

    const nuevoPedido = {
      numeroOrden: document.getElementById("orden").value,
      //principal: document.getElementById("principal").value.split(",").map(item => item.trim()),
      //complementos: document.getElementById("complementos").value.split(",").map(item => item.trim()).filter(Boolean),
      //postre: document.getElementById("postre").value.split(",").map(item => item.trim()).filter(Boolean),
      principal: parsearProductos(document.getElementById("principal").value, especificacionesG),
      complementos: parsearProductos(document.getElementById("complementos").value, especificacionesG),
      postre: parsearProductos(document.getElementById("postre").value, especificacionesG),
      canal: document.getElementById("canal").value || "Mostrador",
      especificaciones: especificacionesG,//document.getElementById("especificaciones").value.split(",").map(item => item.trim()).filter(Boolean),
      estado: "Pendiente"
    };
  
    //pedidos.push(nuevoPedido); // Lo agregamos a la lista
    //cargarPedidos(); // Y lo mostramos
    //this.reset(); // Limpiamos el formulario
    // Enviar el nuevo pedido al backend
    fetch("http://localhost:8000/pedidos/", {
      method: "POST",
      headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(nuevoPedido)
  })
  .then(response => {
  if (!response.ok) {
    throw new Error("Error al guardar el pedido.");
  }
  return response.json();
})
.then(data => {
  console.log("Pedido guardado:", data);
  // Volver a cargar todos los pedidos actualizados
  return fetch("http://localhost:8000/pedidos");
})
.then(response => response.json())
.then(data => {
  console.log("Pedidos recibidos desde la API:", data);
  pedidos = data;
  cargarPedidos();
  this.reset(); //limpiamos formulario
})
.catch(error => {
  console.error("Error:", error);
});

//this.reset(); // Limpiamos el formulario
  });