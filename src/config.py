# src/configuracion.py
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

# Definición de tipos para las funciones personalizadas:
# Todas reciben arreglos de NumPy (para aprovechar la vectorización) y devuelven arreglos.
FuncionCosto = Callable[[np.ndarray], np.ndarray]      # f(q) -> costo total
FuncionDemanda = Callable[[np.ndarray], np.ndarray]    # f(p) -> cantidad demandada
FuncionOferta = Callable[[np.ndarray], np.ndarray]     # f(p) -> cantidad ofrecida (si aplica heurística estática)

# --- CURVAS PREDETERMINADAS (Por defecto) ---

def costo_particular(q: np.ndarray) -> np.ndarray:
    """Función de costo estándar: CF + CVM * q"""
    costo_fijo = 10.0
    costo_marginal = 2.0
    # np.where asegura que si la producción es cero, solo se pague el costo fijo
    return np.where(q > 0, costo_fijo + costo_marginal * q, 0.0)

def demanda_particular(p: np.ndarray) -> np.ndarray:
    """Función de demanda estándar: q = p ^ epsilon_d"""
    epsilon_d = -1.2
    # Evitamos división por cero o raíces negativas asegurando un precio mínimo
    p_seguro = np.maximum(p, 1e-4)
    return p_seguro ** epsilon_d


# --- CONFIGURACIÓN ESTRUCTURADA ---

@dataclass(frozen=True)
class ParamsDimensiones:
    N: int = 500             # Consumidores
    M: int = 50              # Firmas
    t_max: int = 1000        # Periodos

@dataclass(frozen=True)
class ParamsInstitucionales:
    L: int = 5               # Fricción de búsqueda (Núcleo Hayekiano)
    costo_merma: float = 0.0 

@dataclass(frozen=True)
class ParamsConsumidor:
    # Inyección de la curva de demanda personalizada
    curva_demanda: FuncionDemanda = demanda_particular
    presupuesto_medio: float = 100.0
    presupuesto_sigma: float = 0.2

@dataclass(frozen=True)
class ParamsProductor:
    # Inyección de la curva de costos personalizada
    curva_costo: FuncionCosto = costo_particular
    capacidad_max_produccion: float = 150.0
    
    # Parámetros de las heurísticas adaptativas
    factor_ajuste_precio: float = 0.05
    factor_ajuste_produccion: float = 0.10
    tasa_depreciacion_stock: float = 0.20

@dataclass(frozen=True)
class ConfigGlobal:
    dimensiones: ParamsDimensiones = field(default_factory=ParamsDimensiones)
    instituciones: ParamsInstitucionales = field(default_factory=ParamsInstitucionales)
    consumidores: ParamsConsumidor = field(default_factory=ParamsConsumidor)
    productores: ParamsProductor = field(default_factory=ParamsProductor)
    seed: int = 42

    def obtener_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)