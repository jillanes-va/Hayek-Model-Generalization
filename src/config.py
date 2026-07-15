# src/configuracion.py
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

# Definición de tipos para las funciones personalizadas:
# Todas reciben arreglos de NumPy (para aprovechar la vectorización) y devuelven arreglos.
FuncionCosto = Callable[..., np.ndarray]      # f(q, **kwargs) -> costo total
FuncionDemanda = Callable[..., np.ndarray]    # f(p, **kwargs) -> cantidad demandada
FuncionOferta = Callable[..., np.ndarray]     # f(p, **kwargs) -> cantidad ofrecida

# --- CURVAS PREDETERMINADAS (Por defecto) ---

def costo_particular(q: np.ndarray) -> np.ndarray:
    """Función de costo estándar: CF + CVM * q**2"""
    costo_fijo = 10.0
    costo_marginal = 2.0
    # np.where asegura que si la producción es cero, solo se pague el costo fijo
    return np.where(q > 0, costo_fijo + costo_marginal * q**2, 0.0)

def demanda_particular(p: np.ndarray, **kwargs) -> np.ndarray:
    """Función de demanda estándar: q = p ^ epsilon_d"""
    print("¡ALERTA: Entrando a la función particular!") # <-- Añade esto
    epsilon_d = -1.2
    # Evitamos división por cero o raíces negativas asegurando un precio mínimo
    p_seguro = np.maximum(p, 1e-4)
    return p_seguro ** epsilon_d

def oferta_particular(p: np.ndarray) -> np.ndarray:
    """Función de oferta estándar: q = p / (2 * c) para un costo marginal lineal"""
    c = 0.5
    # CMg = 2 * c * q  =>  q = P / (2 * c)
    produccion_optima = p / (2 * c)
    return np.maximum(produccion_optima, 0.0) # Evita producciones negativas


# --- CONFIGURACIÓN ESTRUCTURADA ---

@dataclass(frozen=True)
class ParamsDimensiones:
    N: int = 10000           # Consumidores
    M: int = 100             # Firmas
    t_max: int = 1000        # Periodos
    J: int = 3               # Cadenas de Markov independientes

@dataclass(frozen=True)
class ParamsInstitucionales:
    L: int = 5               # Fricción de búsqueda (Núcleo Hayekiano)
    deepest_search: int = 3  # Profundidad máxima de búsqueda de cada consumidor
    R: int = 50              # Ventana de periodos hacia atrás para evaluar la varianza
    G: float = 1.1           # Umbral estricto de convergencia

@dataclass(frozen=True)
class ParamsConsumidor:
    # Inyección de la curva de demanda personalizada
    curva_demanda: FuncionDemanda = demanda_particular

@dataclass(frozen=True)
class ParamsProductor:
    # Inyección de la curva de costos personalizada
    curva_costo: FuncionCosto = costo_particular
    curva_oferta: FuncionOferta = oferta_particular
    capacidad_max_produccion: float = 150.0

    #Memoria y exploración estocástica
    T_memoria: int = 5           # Periodos de historia que la firma recuerda
    sigma_temblor: float = 0.05  # Desviación estándar del ruido exploratorio (temblor)

@dataclass(frozen=True)
class ConfigGlobal:
    dimensiones: ParamsDimensiones = field(default_factory=ParamsDimensiones)
    instituciones: ParamsInstitucionales = field(default_factory=ParamsInstitucionales)
    consumidores: ParamsConsumidor = field(default_factory=ParamsConsumidor)
    productores: ParamsProductor = field(default_factory=ParamsProductor)
    seed: int = 42

    precios_iniciales: Optional[np.ndarray] = None

    def obtener_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)