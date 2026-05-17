import logging
import os
from datetime import datetime

def setup_logger(name="LegoDetection"):
    """
    Complete logic for a dual-stream logger (Console + File).
    """
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique filename based on time
    log_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger is already initialized
    if not logger.handlers:
        # 1. File Handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 2. Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Logic for easy importing
log = setup_logger()