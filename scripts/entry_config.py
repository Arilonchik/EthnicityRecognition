import torch
from FER.configs import config_resnet34

WORK_DIR = "input/"
DONE_DIR = "output/"
HOST = "127.0.0.1"
MODEL_CONFIG = config_resnet34
DEVICE = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

