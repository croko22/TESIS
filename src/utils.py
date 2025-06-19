# src/utils.py

import random
import numpy as np
import torch
import logging
import os

def set_seed(seed: int = 42):
    """
    Establece la semilla para la reproducibilidad en PyTorch, NumPy y Python.

    Args:
        seed (int): La semilla a establecer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # para todas las GPUs si usas varias
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(name='qg_logger', level=logging.INFO, log_file=None):
    """
    Configura un logger para tu aplicación.

    Args:
        name (str): Nombre del logger.
        level (int): Nivel de logging (ej: logging.INFO, logging.DEBUG).
        log_file (str, optional): Ruta al archivo de log. Si es None, loguea en la consola.

    Returns:
        logging.Logger: El objeto logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo (opcional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de cómo establecer una semilla
    set_seed(123)
    print(f"Semilla establecida a {123}")
    # Puedes probar a generar números aleatorios aquí para verificar

    # Ejemplo de cómo configurar un logger
    logger = setup_logging(log_file="app.log")
    logger.info("Este es un mensaje informativo.")
    logger.warning("Este es un mensaje de advertencia.")
    logger.error("Este es un mensaje de error.")