"""Miscellaneous utility functions."""

import random
import numpy as np
import torch


def seed(value=42):
    """Set random seed for everything.

    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


def map_dict_to_obj(dic):
    result_dic = {}
    if dic is not None:
        for k, v in dic.items():
            if isinstance(v, dict):
                result_dic[k] = map_dict_to_obj(v)
            else:
                try:
                    obj = configmapper.get_object("params", v)
                    result_dic[k] = obj
                except:
                    result_dic[k] = v
    return result_dic
