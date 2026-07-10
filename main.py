# main.py
from src.config import ConfigGlobal
from src.market import Mercado
from src.statistics import calcular_gelman_rubin, exportar_datos_simulacion

def correr_simulacion():
    print("Iniciando simulación Hayekiana...")
    
    config = ConfigGlobal()
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
            
            if r_hat < G:
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