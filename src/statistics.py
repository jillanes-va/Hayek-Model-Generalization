# src/estadisticas.py
import pandas as pd
import numpy as np
from pathlib import Path
from market import Mercado

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
    return float(np.sqrt(V_hat / W))

def exportar_datos_simulacion(mercado: Mercado, nombre_experimento: str = "simulacion_base"):
    """
    Toma los historiales del mercado y los guarda en data/raw/
    """
    # 1. Crear el directorio si no existe
    ruta_salida = Path("data/raw")
    ruta_salida.mkdir(parents=True, exist_ok=True)
    
    # 2. Guardar la Macro (Convergencia y Gelman-Rubin)
    # historial_precios_macro tiene forma (J_cadenas, t_periodos)
    df_macro = pd.DataFrame(
        mercado.historial_precios_macro.T, # Transponemos para que las columnas sean las cadenas
        columns=[f"Cadena_{j+1}" for j in range(mercado.J)]
    )
    df_macro.index.name = "Periodo"
    df_macro.to_parquet(ruta_salida / f"{nombre_experimento}_macro.parquet")
    
    # 3. Guardar la Micro (Datos crudos si los necesitas)
    # Por ejemplo, si en tu mercado añadiste una lista de 'Transaccion' llamada 'registro_transacciones'
    if hasattr(mercado, 'registro_transacciones') and mercado.registro_transacciones:
        df_micro = pd.DataFrame(mercado.registro_transacciones)
        df_micro.to_parquet(ruta_salida / f"{nombre_experimento}_micro_tx.parquet")
        
    print(f"Datos exportados exitosamente en: {ruta_salida.absolute()}")