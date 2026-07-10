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
        self.J = config.dimensiones.J
        self.R = config.instituciones.R
        self.L = config.instituciones.L
        self.G = config.instituciones.G
        self.deepest_search = config.instituciones.deepest_search
        
        
        self.N = config.dimensiones.N
        self.M = config.dimensiones.M
        
        # Almacenamos el historial de precios promedio de cada cadena para Gelman-Rubin
        # Forma: (J cadenas, 0 periodos iniciales) - crecerá dinámicamente
        self.historial_precios_macro = np.zeros((self.J, 0))
        
        # Instanciamos los agentes para cada cadena (aislando las semillas aleatorias)
        self.cadenas_consumidores = []
        self.cadenas_productores = []

        self.registro_transacciones = []
        
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
            #cons.rng = rng_cadena          # No es necesario para consumidores si no hay aleatoriedad interna
            
            self.cadenas_productores.append(prod)
            self.cadenas_consumidores.append(cons)

    def ejecutar_periodo(self, t: int) -> None:
        """
        Ejecuta un periodo completo para las J cadenas.
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
                    
                    # --- RECOLECCIÓN DE DATOS CRUDOS ---
                    tx = Transaccion(
                        periodo=t,
                        id_cadena=j,
                        id_consumidor=int(i),
                        id_firma=int(id_firma),
                        precio=float(precio_transaccion),
                        cantidad=float(q_transada)
                    )
                    self.registro_transacciones.append(tx)
                    
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