import logging
from src.config import Config


def setup_logger():
    logger = logging.getLogger("UBEA-CyberGuard")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger