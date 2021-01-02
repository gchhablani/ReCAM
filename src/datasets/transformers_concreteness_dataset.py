import os
import random
import math
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.mapper import configmapper


@configmapper.map("datasets", "transformers_concreteness")
class TransformersConcretenessDataset(Dataset):
    """Implements the ConcretenessDataset for Concreteness Ratings.

    Args:
        config (src.utils.configuration.Config): The configuration for the dataset class

    Methods:
        __init__(config): constructor for preprocessing and initialising the dataset
        __len__(): Number of samples in the dataset
        __getitem__(idx): Returns the sample corresponding to index=idx in the dataset
        custom_collate_fn(batch): Returns the batch that will be the input to the model
    """

    def __init__(
        self,
        config,
        tokenizer,
    ):

        """
        Construct the TransformersConcretenessDataset.

        Args:
            config (src.utils.configuration.Config): Configuration for the dataset class
        """
        super(ConcretenessDataset, self).__init__()
        # self.data = pickle.load(open(data_path, "rb"))
        self.config = config
        self.csv_file = pd.read_csv(
            self.config.file_path, error_bad_lines=False, delimiter="\t"
        )

        self.data = self.csv_file.to_dict(orient="records")

        self.tokenizer = tokenizer

    def __len__(self):

        """
        Return the length of data.

        Returns:
            shape (int): length of the data
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Get processed text and label at a particular index.

        Args:
            idx (int): The index of the record to be fetched.
        Returns:
            inputs (dict): inputs with keys "input_ids", "token_type_ids", "attention_mask", "labels"
        """

        text = self.data[idx][self.config.text_cols]
        score = self.data[idx][self.config.label_cols]

        text = str(text).lower().split("_")
        text = " ".join(text)

        inputs = self.tokenizer(text=text, max_length=512, truncation=True)
        inputs["labels"] = score

        return inputs

    def custom_collate_fn(self, batch):
        """
        Get a tuple of input dictionaries and returns the batched inputs dictionary.

        Args:
            batch (tuple of dict)
        Returns:
            inputs (dict of torch tensors): keys "input_ids", "token_type_ids", "attention_mask", "labels"
        """

        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        max_len = 0
        inputs = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in batch:
            max_len = max(max_len, len(sample["input_ids"].tolist()[0]))

            for key in sample:
                if key != "labels":
                    inputs[key].append(sample[key].tolist()[0].copy())
                else:
                    inputs[key].append(sample[key])

        # Applying padding to articles relative to the batch
        for i in range(len(inputs["input_ids"])):

            inputs["input_ids"][i] = inputs["input_ids"][i] + [pad_id] * (
                max_len - len(inputs["input_ids"][i])
            )
            inputs["attention_mask"][i] = inputs["attention_mask"][i] + [0] * (
                max_len - len(inputs["attention_mask"][i])
            )
            inputs["token_type_ids"][i] = inputs["token_type_ids"][i] + [0] * (
                max_len - len(inputs["token_type_ids"][i])
            )

        for key in inputs:
            inputs[key] = torch.tensor(inputs[key])
        # to conform with our pattern, do this
        return inputs, inputs["labels"]
