# src/logs.py
import logging
from pathlib import Path

def configurar_logger():
    ruta_logs = Path("data/logs")
    ruta_logs.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=ruta_logs / "mercado.log",
        level=logging.INFO,
        format="%(asctime)s - Cadena %(threadName)s - [%(levelname)s]: %(message)s"
    )
    return logging.getLogger("ABM_Hayek")