# src/agents.py
import numpy as np
from src.config import ConfigGlobal
from typing import Optional

class SectorConsumidor:
    """
    Representa agregadamente a los N consumidores del mercado.
    Su única responsabilidad es evaluar su función de demanda para las señales de precio.
    """
    
    def __init__(self, config: ConfigGlobal):
        self.config = config
        self.N = config.dimensiones.N

    def calcular_demanda(self, precios_enfrentados: np.ndarray) -> np.ndarray:
        """
        Calcula la cantidad demandada puramente en base a la curva teórica.
        
        Parámetros:
        - precios_enfrentados: Arreglo 1D de tamaño N o 1.
                               
        Retorna:
        - Arreglo 1D de tamaño N con las cantidades demandadas.
        """
        # --- VALIDACIÓN ESTRUCTURAL (Para agilizar el debugging del mercado) ---
        assert isinstance(precios_enfrentados, np.ndarray), "Los precios deben ser un arreglo de NumPy."
        assert precios_enfrentados.ndim == 1, "El vector de precios debe ser 1D."
        assert precios_enfrentados.shape[0] in (self.N, 1), f"Se esperaba dimensión {self.N} o 1."

        # --- EVALUACIÓN PURA ---
        # Evalúa estrictamente la función inyectada desde config.py
        demanda = self.config.consumidores.curva_demanda(precios_enfrentados)

        return demanda


class SectorProductor:
    """
    Representa a las M firmas del mercado.
    Planifica la producción buscando el precio óptimo histórico más un temblor lognormal.
    El inventario es 100% perecedero (no se acumula entre periodos).
    """
    
    def __init__(self, config: ConfigGlobal, precios_iniciales: np.ndarray):
        self.config = config
        self.M = config.dimensiones.M
        self.rng = self.config.obtener_rng()
        
        # Parámetros de la heurística (asumimos que los agregarás al config.py)
        self.T = config.productores.T_memoria
        self.sigma = config.productores.sigma_temblor
        
        # --- VALIDACIÓN INICIAL ---
        assert isinstance(precios_iniciales, np.ndarray) and precios_iniciales.shape == (self.M,)
        
        # --- ESTADO DE LA OFERTA ---
        self.precios_actuales = precios_iniciales.copy()
        self.produccion_actual = np.zeros(self.M)
        
        # --- MEMORIA DEL AGENTE (Racionalidad Limitada) ---
        # Matrices 2D de tamaño (M, T) para guardar el historial de cada firma
        self.memoria_precios = np.zeros((self.M, self.T))
        self.memoria_beneficios = np.full((self.M, self.T), -np.inf) # -inf para que los ceros no ganen por defecto
        self.periodos_registrados = np.zeros(self.M, dtype=int)

    def registrar_resultados(self, precios_cobrados: np.ndarray, beneficios_obtenidos: np.ndarray):
        """
        Guarda los resultados del periodo que acaba de terminar en la memoria de la firma.
        Funciona como una cola circular (desplaza los datos antiguos).
        """
        # Desplazamos las columnas hacia la derecha (perdemos el dato más antiguo en índice T-1)
        self.memoria_precios = np.roll(self.memoria_precios, shift=1, axis=1)
        self.memoria_beneficios = np.roll(self.memoria_beneficios, shift=1, axis=1)
        
        # Insertamos el resultado reciente en la primera columna (índice 0)
        self.memoria_precios[:, 0] = precios_cobrados
        self.memoria_beneficios[:, 0] = beneficios_obtenidos
        
        # Actualizamos el contador de experiencia (con tope en T)
        self.periodos_registrados = np.minimum(self.periodos_registrados + 1, self.T)

    def planificar_produccion(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Decide el precio y la producción para el periodo t+1.
        
        Retorna:
        - (precios_t1, produccion_t1)
        """
        # 1. ENCONTRAR P_BEST (El que generó mayor ganancia en el historial)
        # argmax(axis=1) devuelve el índice de la columna con el mayor beneficio para cada firma
        indices_mejores = np.argmax(self.memoria_beneficios, axis=1)
        
        # Extraemos el precio correspondiente a ese índice (Advanced Indexing de NumPy)
        p_best = self.memoria_precios[np.arange(self.M), indices_mejores]
        
        # 2. CONDICIÓN DE MEMORIA CORTA
        # Si la firma tiene menos de T registros, no confía en P_best y mantiene su precio actual
        mascara_novatos = self.periodos_registrados < self.T
        p_base = np.where(mascara_novatos, self.precios_actuales, p_best)
        
        # 3. TEMBLOR DE LA MANO (Exploración Estocástica)
        # Fórmula: P_t+1 = P_base * exp(epsilon - sigma^2 / 2)
        epsilon = self.rng.normal(loc=0.0, scale=self.sigma, size=self.M)
        factor_temblor = np.exp(epsilon - (self.sigma**2) / 2.0)
        
        self.precios_actuales = p_base * factor_temblor
        
        # 4. DECISIÓN DE PRODUCCIÓN NEOCLÁSICA (Deducida de la función de costos)
        # Inyectas la curva_oferta en config.py que resuelve Q = CMg^{-1}(P)
        self.produccion_actual = self.config.productores.curva_oferta(self.precios_actuales)
        
        # Validamos que no se intente producir cantidades negativas
        self.produccion_actual = np.maximum(self.produccion_actual, 0.0)
        
        # Como el inventario no se conserva, la producción es directamente la oferta disponible
        return self.precios_actuales, self.produccion_actual

    def calcular_costos_totales(self) -> np.ndarray:
        """
        Calcula el costo incurrido en el periodo en base a lo producido.
        Se evalúa la curva de costos real inyectada.
        """
        return self.config.productores.curva_costo(self.produccion_actual)