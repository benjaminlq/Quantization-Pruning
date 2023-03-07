import os
import os.path as osp

HOME_PATH = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_PATH = osp.join(HOME_PATH, "data")
MODEL_PATH = osp.join(HOME_PATH, "models")

NUM_WORKERS = os.cpu_count()