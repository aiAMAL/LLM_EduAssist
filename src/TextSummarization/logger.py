import sys
import logging
from pathlib import Path
from datetime import datetime

# Logging format
LOGGING_FORMAT = '[%(asctime)s]: %(levelname)s: %(module)s: %(message)s'


# Log directory and file setup paths
LOG_DIR = Path('logs')
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILEPATH = LOG_DIR / LOG_FILE

# Create log directory if it does not exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger name
LOGGER_NAME = 'TextSummarization'
logger = logging.getLogger(LOGGER_NAME)
