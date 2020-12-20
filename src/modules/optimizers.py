" Method containing activation functions"
from torch.optim import Adam, AdamW
from src.utils.mapper import configmapper

configmapper.map('optimizers', 'adam')(Adam)
configmapper.map('optimizers', 'adam_w')(AdamW)
