import os
import os.path as osp
import logging
import sys

HOME_PATH = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_PATH = osp.join(HOME_PATH, "data")
MODEL_PATH = osp.join(HOME_PATH, "models")

NUM_WORKERS = os.cpu_count()

### Logging configurations
LOGGER = logging.getLogger(__name__)
LOG_PATH = osp.join(MODEL_PATH, "logs")
if not osp.exists(LOG_PATH):
    os.makedirs(LOG_PATH, exist_ok=True)

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(
    filename=osp.join(LOG_PATH, "history.log")
)

formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)