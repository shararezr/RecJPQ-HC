import importlib
from config import BERT4RecExperimentConfig
from bert4rec import BERT4RecPytorchRecommender
import torch


def load_config(config_file: str) -> BERT4RecExperimentConfig:
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def build_model(config: BERT4RecExperimentConfig):
    return BERT4RecPytorchRecommender(config)


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device
