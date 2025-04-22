let carrito = {
    1: 0,
    2: 0
  };
  
  const productos = {
    1: { nombre: "McTrío Mediano BBQ", precio: 99, imagen: "hambur1.png" },
    2: { nombre: "McTrío con Queso", precio: 89, imagen: "hambur2.png" }
  };
  
  function cambiarCantidad(id, cambio) {
    const cantidadElem = document.getElementById(`cantidad${id}`);
    const menosBtn = document.getElementById(`btnMenos${id}`);
  
    carrito[id] = Math.max(0, carrito[id] + cambio);
    cantidadElem.textContent = carrito[id];
  
    menosBtn.style.display = carrito[id] > 0 ? "inline-block" : "none";
  
    actualizarCarritoVisual();
  }
  
  function actualizarCarritoVisual() {
    const total = Object.values(carrito).reduce((sum, val) => sum + val, 0);
    document.querySelectorAll(".cart-icon, .ver-pedido").forEach(elem => {
      elem.setAttribute("data-total", total);
    });
  }
  
  function mostrarPedido() {
    const panel = document.getElementById("carritoPanel");
    const contenido = document.getElementById("carritoContenido");
    contenido.innerHTML = "";
  
    let total = 0;
  
    Object.entries(carrito).forEach(([id, cantidad]) => {
      if (cantidad > 0) {
        const prod = productos[id];
        const subtotal = prod.precio * cantidad;
        total += subtotal;
  
        const item = document.createElement("div");
        item.className = "carrito-item";
        item.innerHTML = `
          <img src="${prod.imagen}" alt="${prod.nombre}">
          <div>
            <p><strong>${prod.nombre}</strong></p>
            <p>Cantidad: ${cantidad}</p>
            <p>Subtotal: $${subtotal}</p>
          </div>
        `;
        contenido.appendChild(item);
      }
    });
  
    document.getElementById("totalPrecio").textContent = `$${total}`;
    panel.style.right = "0";
  }
  
  function cerrarPedido() {
    document.getElementById("carritoPanel").style.right = "-100%";
  }
  