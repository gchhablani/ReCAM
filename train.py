"""Train File."""
## Imports
import argparse
import itertools
import numpy as np
import copy
import torch
import torch.nn as nn
from src.utils.misc import seed, generate_grid_search_configs
from src.utils.configuration import Config
from src.datasets.concreteness_dataset import ConcretenessDataset
from src.trainers.forty import FortyTrainer
from src.modules.preprocessors import *
from src.utils.mapper import configmapper
from src.models.two_layer_nn import TwoLayerNN

## Config
parser = argparse.ArgumentParser(prog="train.py", description="Train a model.")
parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
    default="./configs/models/forty/default.yaml",
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
    default="./configs/trainers/forty/train.yaml",
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
    default="./configs/datasets/forty/default.yaml",
)
parser.add_argument(
    "--grid_search",
    action="store_true",
    help="Whether to do a grid_search",
    default=False,
)
### Update Tips : Can provide more options to the user.
### Can also provide multiple verbosity levels.

args = parser.parse_args()
# print(vars(args))
model_config = Config(path=args.model)
train_config = Config(path=args.train)
data_config = Config(path=args.data)
grid_search = args.grid_search

# verbose = args.verbose

## Seed
seed(42)

# Preprocessor, Dataset, Model
preprocessor = configmapper.get_object(
    "preprocessors", data_config.main.preprocessor.name
)(data_config)

model, train_data, val_data = preprocessor.preprocess(model_config, data_config)

if grid_search:

    train_configs = generate_grid_search_configs(train_config, train_config.grid_search)
    print(f"Total Configurations Generated: {len(train_configs)}")
    for train_config in train_configs:
        print(train_config)
        # Trainer
        trainer = configmapper.get_object("trainers", train_config.name)(train_config)

        ## Train
        trainer.train(model, train_data, val_data)

else:

    ## Trainer
    trainer = configmapper.get_object("trainers", train_config.name)(train_config)

    ## Train
    trainer.train(model, train_data, val_data)