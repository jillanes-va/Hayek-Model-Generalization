# src/market.py
import numpy as np
from typing import List, Dict, Tuple
from config import ConfigGlobal
from agents import SectorConsumidor, SectorProductor
from data_structures import EstadisticasPeriodo, Transaccion

class Mercado:
    """
    Orquesta el Entorno Institucional.
    Ejecuta J cadenas paralelas y maneja el emparejamiento descentralizado.
    """
    
    def __init__(self, config: ConfigGlobal):
        self.config = config
        self.J = config.instituciones.J
        self.L = config.instituciones.L
        self.deepest_search = config.instituciones.deepest_search
        
        self.N = config.dimensiones.N
        self.M = config.dimensiones.M
        
        # Almacenamos el historial de precios promedio de cada cadena para Gelman-Rubin
        # Forma: (J cadenas, 0 periodos iniciales) - crecerá dinámicamente
        self.historial_precios_macro = np.zeros((self.J, 0))
        
        # Instanciamos los agentes para cada cadena (aislando las semillas aleatorias)
        self.cadenas_consumidores = []
        self.cadenas_productores = []
        
        for j in range(self.J):
            # Semilla independiente para cada cadena: config.seed + j
            rng_cadena = np.random.default_rng(self.config.seed + j)
            
            # Inicialización de precios aleatoria para el periodo 0
            precios_init = rng_cadena.uniform(5.0, 25.0, size=self.M)
            
            # Instanciamos los sectores pasándoles el config
            # (Asumimos que internamente usan el config para leer sus curvas)
            prod = SectorProductor(self.config, precios_iniciales=precios_init)
            cons = SectorConsumidor(self.config)
            
            # Forzamos a que usen el RNG de su respectiva cadena
            prod.rng = rng_cadena
            cons.rng = rng_cadena
            
            self.cadenas_productores.append(prod)
            self.cadenas_consumidores.append(cons)

    def ejecutar_periodo(self, t: int) -> bool:
        """
        Ejecuta un periodo completo para las J cadenas.
        Retorna True si el sistema convergió según Gelman-Rubin, False en caso contrario.
        """
        precios_promedio_cadenas = np.zeros(self.J)

        for j in range(self.J):
            cons = self.cadenas_consumidores[j]
            prod = self.cadenas_productores[j]
            rng = prod.rng # Usamos el RNG de la cadena actual
            
            # 1. Productores fijan precios y producción inicial
            precios_oferta, stock_disponible = prod.planificar_produccion()
            
            # --- FASE DE DESCUBRIMIENTO (VECTORIZADA) ---
            # Un truco ultra-rápido para hacer que N consumidores elijan L firmas sin reemplazo
            ruido = rng.random((self.N, self.M))
            muestras_L = np.argsort(ruido, axis=1)[:, :self.L]
            
            # Extraemos los precios de esas muestras (Matriz N x L)
            precios_muestras = precios_oferta[muestras_L]
            
            # Ordenamos las muestras de menor a mayor precio para cada consumidor
            orden_precios = np.argsort(precios_muestras, axis=1)
            firmas_ordenadas = np.take_along_axis(muestras_L, orden_precios, axis=1)
            
            # Acotamos la búsqueda a la profundidad establecida
            firmas_a_visitar = firmas_ordenadas[:, :self.deepest_search]
            
            # El consumidor evalúa su demanda en base al mejor precio encontrado (índice 0)
            mejores_precios_vistos = np.min(precios_muestras, axis=1)
            demanda_restante = cons.calcular_demanda(mejores_precios_vistos)
            
            # --- FASE DE ASIGNACIÓN SECUENCIAL ---
            # Orden aleatorio de llegada al mercado para no favorecer a los primeros índices
            orden_llegada = rng.permutation(self.N)
            
            # Registros del periodo para la cadena j
            ingresos_firmas = np.zeros(self.M)
            q_vendida_firmas = np.zeros(self.M)
            # Aquí podrías instanciar tu lista de Transacciones Raw para el log profundo
            
            for i in orden_llegada:
                if demanda_restante[i] <= 0:
                    continue
                    
                # Recorre las opciones más baratas acotadas por deepest_search
                for id_firma in firmas_a_visitar[i]:
                    stock = stock_disponible[id_firma]
                    if stock <= 0:
                        continue # La firma agotó stock, pasa a la siguiente
                        
                    precio_transaccion = precios_oferta[id_firma]
                    q_transada = min(demanda_restante[i], stock)
                    
                    # Actualización de estados
                    stock_disponible[id_firma] -= q_transada
                    demanda_restante[i] -= q_transada
                    
                    # Contabilidad de la firma
                    q_vendida_firmas[id_firma] += q_transada
                    ingresos_firmas[id_firma] += (q_transada * precio_transaccion)
                    
                    # (Aquí guardarías la Transaccion en tu bitácora cruda)
                    
                    if demanda_restante[i] <= 0:
                        break # Consumidor sació su demanda, sale del mercado
            
            # 3. Productores evalúan sus beneficios y aprenden
            # Beneficio = Ingresos - Costos Totales de Producción
            costos_totales = prod.calcular_costos_totales()
            beneficios = ingresos_firmas - costos_totales
            
            prod.registrar_resultados(precios_oferta, beneficios)
            
            # 4. Cálculo del macro-estado de la cadena j (Precio Promedio Ponderado por volumen)
            volumen_total_j = np.sum(q_vendida_firmas)
            if volumen_total_j > 0:
                precio_prom_j = np.sum(ingresos_firmas) / volumen_total_j
            else:
                precio_prom_j = np.mean(precios_oferta) # Si no hay transacciones
                
            precios_promedio_cadenas[j] = precio_prom_j

        # --- EVALUACIÓN DE CONVERGENCIA DEL SISTEMA ---
        # Anexamos los precios del periodo actual al historial global
        self.historial_precios_macro = np.column_stack((self.historial_precios_macro, precios_promedio_cadenas))
        
        # Calculamos Gelman-Rubin si ya tenemos suficientes periodos (R)
        if t >= self.config.instituciones.R:
            r_hat = self.calcular_gelman_rubin()
            if r_hat < self.config.instituciones.G:
                return True # El sistema convergió
                
        return False

    def calcular_gelman_rubin(self) -> float:
        """
        Calcula el estadístico R-hat para el precio promedio entre las J cadenas,
        utilizando los últimos R periodos del historial.
        """
        R_window = self.config.instituciones.R
        J = self.J
        
        # Extraemos solo la matriz de los últimos R periodos. Forma: (J, R)
        datos = self.historial_precios_macro[:, -R_window:]
        
        # Media de cada cadena (sobre el tiempo)
        medias_cadenas = np.mean(datos, axis=1)
        
        # Media global (sobre las cadenas y el tiempo)
        media_global = np.mean(medias_cadenas)
        
        # Varianza Entre-cadenas (Between-chain variance: B)
        B = (R_window / (J - 1)) * np.sum((medias_cadenas - media_global)**2)
        
        # Varianza Intra-cadenas (Within-chain variance: W)
        varianzas_internas = np.var(datos, axis=1, ddof=1)
        W = np.mean(varianzas_internas)
        
        if W == 0.0:
            return 1.0 # Evita división por cero si el sistema se congeló en un valor exacto
            
        # Estimación de la varianza total del sistema
        V_hat = ((R_window - 1) / R_window) * W + (1 / R_window) * B
        
        # Estadístico R-hat
        r_hat = np.sqrt(V_hat / W)
        return float(r_hat)