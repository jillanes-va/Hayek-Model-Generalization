# main.py
from src.config import ConfigGlobal
from src.market import Mercado
from src.statistics import exportar_datos_simulacion

def correr_simulacion():
    print("Iniciando simulación Hayekiana...")
    
    # 1. Inicializamos los parámetros
    config = ConfigGlobal()
    mercado = Mercado(config)
    
    t_max = config.dimensiones.t_max
    convergio = False
    
    # 2. El bucle del tiempo
    for t in range(t_max):
        # ejecutar_periodo devuelve True si Gelman-Rubin < G
        convergio = mercado.ejecutar_periodo(t)
        
        # Opcional: Imprimir el progreso cada 50 periodos
        if t % 50 == 0:
            print(f"Periodo {t}/{t_max} completado...")
            
        if convergio:
            print(f"\n¡Convergencia alcanzada en el periodo {t}!")
            print(f"Estadístico R-hat final: {mercado.calcular_gelman_rubin():.4f}")
            break
            
    if not convergio:
        print(f"\nSimulación finalizada sin alcanzar convergencia estricta en {t_max} periodos.")

    # 3. Guardar resultados
    exportar_datos_simulacion(mercado, nombre_experimento="prueba_convergencia")

if __name__ == "__main__":
    correr_simulacion()