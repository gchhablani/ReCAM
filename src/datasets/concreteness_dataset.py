"""Implements the ."""

import pandas as pd
from torch.utils.data import Dataset
from src.utils.mapper import configmapper


@configmapper.map("datasets", "concreteness")
class ConcretenessDataset(Dataset):
    """Implement dataset for Concreteness Ratings.

    Attributes:
        config (src.utils.configuration.Config): The configuration for the dataset class
        data (pandas.DataFrame): The dataframe object containing the ratings
        PREPRO (src.utils.Tokenizer): The tokenizer object to be used to tokenize text

    Methods:
        __init__(file_path,tokenizer,split): initialize the dataset
        __len__(): get length of the dataset
        __getitem__(idx): get item at a particular index
    """

    def __init__(self, config, tokenizer):
        """Construct the ConcretenessDataset.

        Args:
            config (src.utils.configuration.Config): Configuration for the dataset class
            tokenizer (src.modules.tokenizer.Tokenizer): Tokenizer for the dataset class
        """
        super(ConcretenessDataset, self).__init__()

        self.config = config
        self.data = pd.read_csv(
            self.config.file_path, error_bad_lines=False, delimiter="\t"
        )[self.config.text_cols + [self.config.label_col,]].dropna()
        self.tokenizer = tokenizer

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
        word = record[self.config.text_cols]
        if self.tokenizer is not None:
            word = self.tokenizer.tokenize(
                word, **self.config.preprocessor.tokenizer.init_vector_params.as_dict()
            )
        if self.config.split == "test":
            return word
        value = record[self.config.label_col]
        # print(word,value)
        return word, value
