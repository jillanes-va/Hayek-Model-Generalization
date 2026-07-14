# experiments/exp_01_friccion_busqueda.py

import sys
from pathlib import Path

#==================Ejecutar en carpeta correcta=======================================
# Buscamos la carpeta que está un nivel arriba de este script (la raíz del proyecto)
raiz_proyecto = str(Path(__file__).resolve().parent.parent)

# La agregamos al principio del sistema de búsqueda de Python si no está presente
if raiz_proyecto not in sys.path:
    sys.path.insert(0, raiz_proyecto)
#=====================================================================================

from src.config import ConfigGlobal
from src.market import Mercado
from src.statistics import calcular_gelman_rubin, exportar_datos_simulacion

import numpy as np
from dataclasses import replace

def demanda_estocastica(p: np.ndarray, rng_activo: np.random.Generator | None = None) -> np.ndarray:
        """Función inyectada desde el experimento."""
        return 10 - 0.5 * p**1.5

def config_particular() -> ConfigGlobal:
    """
    Fábrica: Construye y retorna la configuración exacta de este experimento.
    """

    config = ConfigGlobal()

    #=================ESPACIO DE EXPERIMENTACION===================

    rng = np.random.default_rng(config.seed)
    M = config.dimensiones.M
    J = config.dimensiones.J
    
    matriz_precios = np.zeros((J, M))
    matriz_precios[0, :] = rng.uniform(0.2, 0.8, size=M)  # Cadena 0: Sociedad de precios bajos
    matriz_precios[1, :] = rng.uniform(1.0, 2.0, size=M)  # Cadena 1: Sociedad de precios medios
    matriz_precios[2, :] = rng.uniform(4.0, 5.0, size=M)  # Cadena 2: Sociedad de precios altos

    nuevos_consumidores = replace(config.consumidores, curva_demanda=demanda_estocastica)

    config = replace(config, precios_iniciales=matriz_precios, consumidores=nuevos_consumidores)
    
    return config

def correr_simulacion():
    print("Iniciando simulación Hayekiana...")

    config = config_particular()

    mercado = Mercado(config)
    
    t_max = config.dimensiones.t_max
    R = config.instituciones.R
    G = config.instituciones.G

    
    convergio = False
    
    for t in range(t_max):
        # 1. El mercado opera ciegamente
        mercado.ejecutar_periodo(t)

        if t % 100 == 0:
            print('Iteración:', t, 'de', t_max)
        
        # 2. El orquestador (tú) evalúa la macroestructura
        if t >= R:
            r_hat = calcular_gelman_rubin(mercado.historial_precios_macro, R)
            
            if r_hat < G and t > 400: #Check de convergencia estricta después de un burn-in
                print(f"\nConvergencia alcanzada en el periodo {t}")
                print(f"Estadístico R-hat final: {r_hat:.4f}")
                convergio = True
                break  # Detenemos el reloj
                
    if not convergio:
        print(f"\nSimulación finalizada. No hubo convergencia estricta en {t_max} periodos.")

    # 3. Extraemos los datos crudos del mercado y los mandamos a guardar
    print("Exportando datos a formato Parquet...")
    exportar_datos_simulacion(
        historial_macro=mercado.historial_precios_macro,
        transacciones=mercado.registro_transacciones,
        nombre_experimento="convergencia_hayek_01"
    )

if __name__ == "__main__":
    correr_simulacion()