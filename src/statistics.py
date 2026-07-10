# src/estadisticas.py
import pandas as pd
import numpy as np
from pathlib import Path
from market import Mercado

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