"All criterion functions."
from torch.nn import MSELoss
from src.utils.mapper import configmapper
configmapper.map("losses","mse")(MSELoss)
