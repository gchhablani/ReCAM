"""
Script to run integrated gradients.

Usage:
    $python run_integrated_gradients.py --config ./configs/integrated_gradients_bert_cloze.yaml

"""
import os
import argparse
import pickle as pkl
import pandas as pd


# import itertools
import copy
import numpy as np
import torch
import torch.nn as nn

from src.utils.misc import seed, generate_grid_search_configs
from src.utils.configuration import Config

from src.datasets import *
from src.models import *
from src.trainers import *

from src.modules.preprocessors import *
from src.modules.tokenizers import *
from src.utils.mapper import configmapper
from src.utils.logger import Logger

import os

from src.utils.integrated_gradients import MyIntegratedGradients

# from src.utils.misc import seed

dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="run_integrated_gradients.py",
    description="Run integrated gradients on a model.",
)
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="The configuration for integrated gradients",
)

parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
)

parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
)

args = parser.parse_args()
ig_config = Config(path=args.config)

model_config = Config(path=args.model)
data_config = Config(path=args.data)

# verbose = args.verbose

# Preprocessor, Dataset, Model
preprocessor = configmapper.get_object(
    "preprocessors", data_config.main.preprocessor.name
)(data_config)


model, train_data, val_data = preprocessor.preprocess(model_config, data_config)

tokenizer = configmapper.get_object("tokenizers", (ig_config.checkpoint_path))
model = configmapper.get_object("models", model_config.name).from_pretrained(
    ig_config.checkpoint_path
)
# Initialize BertIntegratedGradients
big = MyIntegratedGradients(ig_config, model, val_data, tokenizer)

print("### Running IG ###")
(
    samples,
    word_importances,
    token_importances,
) = big.get_random_samples_and_importances_across_all_layers(
    n_samples=ig_config.n_samples
)

print("### Saving the Scores ###")
with open(os.path.join(ig_config.store_dir, "samples"), "wb") as out_file:
    pkl.dump(samples, out_file)
with open(os.path.join(ig_config.store_dir, "token_importances"), "wb") as out_file:
    pkl.dump(token_importances, out_file)
with open(os.path.join(ig_config.store_dir, "word_importances"), "wb") as out_file:
    pkl.dump(word_importances, out_file)

print("### Finished ###")
