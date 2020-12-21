import sys

sys.path.append("./")

import os
import random
import math
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# chuck Word2Vec
# import gensim
# from gensim.models.keyedvectors import KeyedVectors
# from gensim.models import Word2Vec


# word_embedding = "data/Imperceptibility/W2V/GoogleNews-vectors-negative300.bin"
# model = KeyedVectors.load_word2vec_format(word_embedding, binary=True)

"""
class ConcretenessDataset(Dataset):
    def __init__(self, csv_file,
        #word_embedding
        ):
        self.csv_file = pd.read_csv(csv_file, error_bad_lines=False, delimiter="\t")
        # self.model = KeyedVectors.load_word2vec_format(word_embedding, binary=True)
        # self.model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # print(self.csv_file.head())
        try:
            s = self.csv_file.iloc[idx]["WORD"].lower().split("_")
            s = " ".join(s)
            # use BERT Tokenizer here

"""


class ConcretenessDataset(object):
    """Implements the ConcretenessDataset for Concreteness Ratings.

    Args:
        data_path: (list) Path of the pickled list
        Format of the list: [{'queryid': query, 'docid': doc,'label': label}, ...]
        where query, doc are strings and label is an integer.
        batch_size: (int)
        tokenizer:
        split: randomly shuffle dataset if split='training'
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        csv_file,
        batch_size,
        tokenizer,
        split="training",
        device=torch.device("cuda"),
    ):
        super(ConcretenessDataset, self).__init__()
        # self.data = pickle.load(open(data_path, "rb"))
        self.csv_file = pd.read_csv(csv_file, error_bad_lines=False, delimiter="\t")
        self.data = self.csv_file.to_dict(orient="records")

        if split != "test":
            np.random.shuffle(self.data)

        self.data_i = 0
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.start = True

    def get_instance(self):
        """Returns one data-point, i.e., one dictionary {'queryid': query, 'docid': doc,'label': label} from the input list"""
        ret = self.data[self.data_i % self.data_size]
        self.data_i += 1
        return ret

    def __len__(self):
        return self.data_size

    def epoch_end(self):
        """Returns true when the end of the epoch is reached, otherwise false"""
        return self.data_i % self.data_size == 0

    def load_batch(self):
        """Takes the required number of data-points (batch_size), computes all the masks and returns the appended inputs+masks"""
        (untokenized_query, label_batch,) = (
            [],
            [],
        )
        while True:
            if not self.start and self.epoch_end():
                self.start = True
                break
            self.start = False
            instance = self.get_instance()

            a = instance["Word"]
            label = instance["Conc.M"]
            print(a)
            print(label)

            query = str(a).lower().split("_")
            query = " ".join(query)

            # print(query)
            # print(label)

            untokenized_query.append(query)
            label_batch.append(label)

            if len(untokenized_query) >= self.batch_size or self.epoch_end():
                # Convert inputs to PyTorch tensors
                inputs = self.tokenizer(
                    text=untokenized_query,
                    # text_pair=untokenized_desc,
                    max_length=512,
                    truncation=True,
                    padding="longest",
                )

                for key in inputs:
                    inputs[key] = torch.tensor(
                        inputs[key], dtype=torch.int64, device=self.device
                    )

                label_tensor = torch.tensor(
                    label_batch, dtype=torch.float32, device=self.device
                )
                # qid_tensor = torch.tensor(qid_batch, device=self.device)
                # docid_tensor = torch.tensor(docid_batch, device=self.device)
                # inputs['labels'] = label_tensor
                return inputs, label_tensor

        return None
