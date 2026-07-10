# src/estructuras.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EstadisticasPeriodo:
    periodo: int
    
    # Vectores de estado de las FIRMAS al cierre del periodo (Dimensión: M)
    precios_firmas: np.ndarray        # Precios fijados por cada firma t
    inventarios_finales: np.ndarray   # Stock remanente de cada firma t
    inventarios_destruidos: np.ndarray # Mermas/vencimientos físicos de cada firma t
    beneficios_firmas: np.ndarray     # Utilidades netas del periodo de cada firma t
    
    # Vectores de estado de los CONSUMIDORES (Dimensión: N)
    cantidades_demandadas: np.ndarray # Lo que querían comprar según su presupuesto
    cantidades_compradas: np.ndarray  # Lo que efectivamente lograron adquirir
    
    # Lista fina de transacciones efectivas (opcional, longitud dinámica)
    # Almacena una lista de tuplas o diccionarios con el matching exacto si lo requieres
    transacciones_raw: list           

    def __post_init__(self):
        """Validación de sanidad estructural y numérica para debugging."""
        assert self.periodo >= 0, "El periodo no puede ser negativo."
        
        # Verificar que no se arrastren dimensiones corruptas
        assert self.precios_firmas.ndim == 1, "precios_firmas debe ser un arreglo 1D."
        assert self.inventarios_finales.shape == self.precios_firmas.shape, "Discrepancia en la dimensión M de firmas."
        assert self.cantidades_compradas.shape == self.cantidades_demandadas.shape, "Discrepancia en la dimensión N de consumidores."
        
        # Alertas tempranas de inconsistencia física
        if np.any(self.inventarios_finales < 0):
            raise ValueError(f"CRÍTICO t={self.periodo}: Se detectaron inventarios negativos.")
        if np.any(self.cantidades_compradas > self.cantidades_demandadas):
            raise ValueError(f"CRÍTICO t={self.periodo}: Consumidores compraron más de su demanda máxima simulada.")