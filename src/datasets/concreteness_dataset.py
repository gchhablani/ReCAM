"""Implements the ."""

import pandas as pd
from torch.utils.data import Dataset


class ConcretenessDataset(Dataset):
    """Implement dataset for Concreteness Ratings.

    Attributes:
        data (pandas.DataFrame): The dataframe object containing the ratings
        tokenizer (src.utils.Tokenizer): The tokenizer object to be used to tokenize text
        split (str): Whether the split is "train", "val" or "test"

    Methods:
        __init__(file_path,tokenizer,split): initialize the dataset
        __len__(): get length of the dataset
        __getitem__(idx): get item at a particular index
    """

    def __init__(self, file_path, tokenizer, split="train", **tokenizer_params):
        """Construct the ConcretenessDataset.

        Args:
            file_path (str): File path for dataset.
            tokenizer (src.utils.Tokenizer): The tokenizer object to be used to tokenize text
            split (str): Whether the split is "train", "val" or "test"
            **tokenizer_params (keyword arguments): Keyword arguments for tokenizer

        """
        super(ConcretenessDataset, self).__init__()

        self.data = pd.read_csv(file_path, error_bad_lines=False, delimiter="\t")[
            ["Word", "Conc.M"]
        ].dropna()
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.split = split

    def __len__(self):
        """Return the length of data.

        Returns:
            shape (int): length of the data
        """
        shape = self.data.shape[0]
        return shape

    def __getitem__(self, idx):
        """Get processed text and label at a particular index.

        Args:
            idx (int): The index of the record to be fetched.

        Returns:
            word (torch.Tensor) if self.split is "test".
            word (torch.Tensor), value(int) is self.split is not "test".
        """
        record = self.data.iloc[idx]
        word = record["Word"]
        if self.tokenizer is not None:
            word = self.tokenizer.tokenize(
                word, fields=["Word"], **self.tokenizer_params
            )
        if self.split == "test":
            return word
        value = record["Conc.M"]
        # print(word,value)
        return word, value
