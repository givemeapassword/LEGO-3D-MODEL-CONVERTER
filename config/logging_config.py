import logging
import os

def setup_logging(log_level=logging.DEBUG, log_dir="logs"):
    """Настраивает логирование для проекта."""
    os.makedirs(log_dir, exist_ok=True)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_filename = os.path.join(log_dir, "app.log")

    logger = logging.getLogger()
    logger.setLevel(log_level)  # Минимальный уровень для логгера — DEBUG

    if logger.handlers:
        logger.handlers.clear()

    # Консольный обработчик (DEBUG и выше)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(console_handler)

    # Файловый обработчик (DEBUG и выше)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    logging.info("Logging configured successfully")