import os
import torch
import numpy as np
import jsonlines
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from src.utils.mapper import configmapper


@configmapper.map("datasets", "cloze_base")
class ClozeDataset(Dataset):
    """Implements dataset for Cloze Style Question-Answering.

    Attributes:
        config (src.utils.configuration.Config): The configuration for the dataset class
        data : The jsonl/pt containing the articles with question and answer

    Methods:
        __init__(file_path,tokenizer,split): initialize the dataset
        __len__(): get length of the dataset
        __getitem__(idx): get item at a particular index
    """

    def __init__(self, config, tokenizer):

        """Construct the ClozeDataset.

        Args:
            config (src.utils.configuration.Config): Configuration for the dataset class
            tokenizer (src.modules.tokenizer.Tokenizer): Tokenizer for the dataset class
        """
        self.config = config
        self.tokenizer = tokenizer

        with jsonlines.open(self.config.file_path) as file:
            self.data = list(file)

        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token
        self.mask_token = self.tokenizer.mask_token

    def __len__(self):
        return len(self.data)

    def _preprocess(self, data):
        """
        Helper Function To preprocess each datapoint before __get_item__ for jsonl files
        """
        article = (
            data["article"].lower()
            + " "
            + data["question"]
            .lower()
            .replace("@placeholder", self.tokenizer.mask_token)
        )
        article = self.tokenizer.encode(article)

        truncated_article = article[-self.config.truncate_length :]
        # Saving the [MASK]'s index
        answer_index = truncated_article.index(self.mask_token_id)

        # Tokenizing Options
        options = [data["option_" + str(i)] for i in range(5)]
        options_tokenized = []
        for i in range(5):
            option = self.tokenizer.encode(options[i])
            options_tokenized.append(option)

        # Storing Answer
        if self.config.split == "test":
            return {
                "article": article,
                "answer_index": answer_index,
                "options": options_tokenized,
            }
        else:
            answer = data["label"]
            return {
                "article": article,
                "answer_index": answer_index,
                "options": options_tokenized,
            }, answer

    def __getitem__(self, idx):
        """
        Gets processed text, masks, options and answers at a particular index
        Args:
            idx (int): The index of the record to be fetched.
        Returns:
            sample(List) -> article,article_masks,options,answer_index,answer
        (all with tokenized input ids)
        """

        if self.config.split == "test":

            data = self._preprocess(self.data[idx])
            data["article_attention_mask"] = [1] * len(data["article"])
            return data

        else:
            data, label = self._preprocess(self.data[idx])
            data["article_attention_mask"] = [1] * len(data["article"])
            return data, label

    def custom_collate_fn(self, batch):
        # Used for batching; Pads all articles and options to equal token lengths
        articles = []
        article_masks = []
        options = []
        answer_indices = []
        answers = []

        max_len = 0
        ops_max_len = 0

        for sample, label in batch:

            articles.append(sample["article"])
            max_len = max(max_len, len(sample["article"]))
            article_masks.append(sample["article_attention_mask"])

            options.append(sample["options"])
            ops_max_len = max(ops_max_len, max([len(i) for i in sample["options"]]))

            answer_indices.append(sample["answer_index"])
            if self.config.split != "test":
                answers.append(label)

        # Applying padding to articles relative to the batch
        for i in range(len(articles)):

            articles[i] = articles[i] + [self.pad_token_id] * (
                max_len - len(articles[i])
            )
            article_masks[i] = article_masks[i] + [self.pad_token_id] * (
                max_len - len(article_masks[i])
            )

        # Applying padding to options relative to the batch
        for i in range(len(options)):
            for j in range(len(options[i])):
                options[i][j] = options[i][j] + [self.pad_token_id] * (
                    ops_max_len - len(options[i][j])
                )

        batch_sample = {
            "articles": torch.LongTensor(articles),
            "article_attention_masks": torch.LongTensor(article_masks),
            "options": torch.LongTensor(options),
            "answer_indices": torch.LongTensor(answer_indices),
        }
        return batch_sample, torch.LongTensor(answers)
