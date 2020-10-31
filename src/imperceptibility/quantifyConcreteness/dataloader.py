import math
import sys
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

word_embedding = "data/Imperceptibility/W2V/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(word_embedding, binary=True)


class ConcretenessDataset(Dataset):
    def __init__(self, csv_file, word_embedding):
        self.csv_file = pd.read_csv(csv_file, error_bad_lines=False, delimiter="\t")
        # self.model = KeyedVectors.load_word2vec_format(word_embedding, binary=True)
        # self.model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # print(self.csv_file.head())
        try:
            s = self.csv_file.iloc[idx]["WORD"].lower().split("_")
            word = np.mean([model[i] for i in s], axis=0)
            # print(word.shape)

            # word = model[str(self.csv_file.iloc[idx]["WORD"]).lower()]
            rating = float(self.csv_file.iloc[idx]["RATING"])
            # print(word)
            # print(rating)
            return word, rating
        except:
            # print("UNK Word in Word2Vec Model")
            return torch.zeros(300), 0
