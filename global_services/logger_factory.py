# global_services/logger_factory.py

import logging
import getpass
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import sys

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s: %(message)s | Name: %(name)s, "
    "File: %(filename)s, Function: %(funcName)s, Line: %(lineno)d"
)

_log_id = None
_logger_instance = None

def create_logger(user: str = "") -> logging.Logger:
    global _log_id, _logger_instance

    if _logger_instance:
        return _logger_instance  # Reuse

    if not user:
        user = getpass.getuser()
    today = datetime.now().strftime("%d-%m-%Y")
    _log_id = f"log__{user}_{today}"

    logs_path = Path("logs")
    logs_path.mkdir(exist_ok=True)
    log_file = logs_path / f"{_log_id}.log"

    logger = logging.getLogger(_log_id)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. File Handler with Daily Rotation
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    # 2. Console Handler (stdout for Docker/Kubernetes)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _logger_instance = logger
    return logger

