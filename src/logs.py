# src/logs.py
import logging

def configurar_auditoria(nombre_archivo: str = "data/raw/simulacion.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (Periodo %(threadName)s): %(message)s",
        handlers=[
            logging.FileHandler(nombre_archivo, mode='w'),
            logging.StreamHandler() # Muestra en consola también
        ]
    )
    return logging.getLogger("ABM_Hayek")