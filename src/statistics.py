# src/estadisticas.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import ConfigGlobal

def calcular_gelman_rubin(historial_precios: np.ndarray, R_window: int) -> float:
    """
    Calcula el estadístico R-hat para evaluar la convergencia hacia el Orden Espontáneo.
    
    Parámetros:
    - historial_precios: Matriz de (J_cadenas, t_periodos).
    - R_window: Ventana de tiempo (profundidad) a evaluar.
    """
    J = historial_precios.shape[0]
    datos = historial_precios[:, -R_window:]
    
    medias_cadenas = np.mean(datos, axis=1)
    media_global = np.mean(medias_cadenas)
    
    # Varianza Entre-cadenas (Between-chain variance)
    B = (R_window / (J - 1)) * np.sum((medias_cadenas - media_global)**2)
    
    # Varianza Intra-cadenas (Within-chain variance)
    varianzas_internas = np.var(datos, axis=1, ddof=1)
    W = np.mean(varianzas_internas)
    
    if W == 0.0:
        return 1.0
        
    V_hat = ((R_window - 1) / R_window) * W + (1 / R_window) * B
    return float(V_hat / W)

def exportar_datos_simulacion(historial_macro: np.ndarray, 
                              transacciones: list, 
                              nombre_experimento: str):
    """Guarda los datos en disco sin saber de dónde vinieron."""
    ruta_salida = Path("data/raw")
    ruta_salida.mkdir(parents=True, exist_ok=True)
    
    # Exportar Macro
    df_macro = pd.DataFrame(
        historial_macro.T,
        columns=[f"Cadena_{j+1}" for j in range(historial_macro.shape[0])]
    )
    df_macro.to_parquet(ruta_salida / f"{nombre_experimento}_macro.parquet")
    
    # Exportar Micro
    if transacciones:
        df_micro = pd.DataFrame(transacciones)
        df_micro.to_parquet(ruta_salida / f"{nombre_experimento}_micro_tx.parquet")


def calcular_equilibrio_marshalliano(config: ConfigGlobal, p_max: float = 30.0, puntos: int = 500) -> Tuple[float, float]:
    """Calcula numéricamente el P* y Q* del equilibrio neoclásico."""
    precios = np.linspace(0.1, p_max, puntos)
    
    q_demanda = config.dimensiones.N * config.consumidores.curva_demanda(precios)
    q_oferta = config.dimensiones.M * config.productores.curva_oferta(precios)
    
    idx = np.argmin(np.abs(q_demanda - q_oferta))
    return precios[idx], q_oferta[idx]