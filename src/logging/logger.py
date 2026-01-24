import logging
import os
from datetime import datetime


LOG_DIR = os.path.join(os.getcwd(),"Logs")
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)


logging.basicConfig(
    filename = LOG_FILE_PATH, 
    level = logging.INFO,
    format = "[%(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)


