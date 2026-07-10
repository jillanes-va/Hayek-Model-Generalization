# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from src.config import ConfigGlobal
from src.statistics import calcular_equilibrio_marshalliano

def graficar_cruz_marshalliana(config: ConfigGlobal, ax=None):
    """
    Genera el gráfico de Oferta y Demanda agregada.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    precios = np.linspace(0.1, 30.0, 500)
    q_demanda = config.dimensiones.N * config.consumidores.curva_demanda(precios)
    q_oferta = config.dimensiones.M * config.productores.curva_oferta(precios)
    
    # Le pedimos la matemática a statistics.py
    p_eq, q_eq = calcular_equilibrio_marshalliano(config)
    
    ax.plot(q_demanda, precios, label='Demanda Agregada', color='blue', linewidth=2)
    ax.plot(q_oferta, precios, label='Oferta Agregada', color='red', linewidth=2)
    
    # Marcador del equilibrio
    ax.scatter(q_eq, p_eq, color='black', s=100, zorder=5, label=f'Eq. Teórico: P*={p_eq:.2f}')
    ax.axhline(p_eq, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(q_eq, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title("Equilibrio Teórico Neoclásico")
    ax.set_xlabel("Cantidad ($Q$)")
    ax.set_ylabel("Precio ($P$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax