import math
import sys
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


class ConcretenessDataset(Dataset):
    def __init__(self, csv_file, word_embedding):
        self.csv_file = pd.read_csv(csv_file, error_bad_lines=False, delimiter="\t")
        self.model = KeyedVectors.load_word2vec_format(word_embedding, binary=True)
        # self.model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # print(self.csv_file.head())
        try:
            word = self.model[str(self.csv_file.iloc[idx]["WORD"])]
            rating = float(self.csv_file.iloc[idx]["RATING"])
            return word, rating
        except:
            print("UNK Word in Word2Vec Model")
