import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from src.utils.mapper import configmapper


@configmapper.map("datasets", "cloze")
class ClozeDataset(Dataset):
    """Implements dataset for Cloze Style Question-Answering.

    Attributes:
        config (src.utils.configuration.Config): The configuration for the dataset class
        data : The jsonl/pt containing the articles with question and answer
        PREPRO (src.utils.Tokenizer): The tokenizer object to be used to tokenize text

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

        if hasattr(self.config, "preprocessed"):
            if self.config.preprocessed == True:
                self.data = torch.load(self.config.file_path)
        else:
            self.config.preprocessed = False
            with open(self.config.file_path) as f:
                self.data = [
                    json.loads(datapoint) for datapoint in f.read().splitlines()
                ]

        self.mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
        self.pad_id = tok.convert_tokens_to_ids("[PAD]")
        # Check if truncate is true but truncate length has not been mentioned
        if hasattr(self.config, "truncate"):
            if not hasattr(self.config, "truncate_length"):
                # Default value 512
                self.config.truncate_length = 512
        else:
            # Make False if Truncate does not Exist
            self.config.truncate = False

    def __len__(self):
        return len(self.data)

    def _preprocess(self, data):
        """
            Helper Function To preprocess each datapoint before __get_item__ for jsonl files
        """
        article = (
            data["article"].lower()
            + " "
            + data["question"].lower().replace("@placeholder", "[MASK]")
        )
        article = self.tokenizer(
            article, return_token_type_ids=False, return_attention_mask=False
        )

        if (
            self.config.truncate
            and len(article["input_ids"]) > self.config.truncate_length
        ):
            article["input_ids"] = article["input_ids"][-self.config.truncate_length :]

        # Saving the [MASK]'s index
        answer_index = article["input_ids"].index(self.mask_id)

        # Tokenizing Options
        options = [data["option_" + str(i)] for i in range(5)]
        options_tokenized = []
        for i in range(5):
            option = self.tokenizer(
                options[i], return_token_type_ids=False, return_attention_mask=False
            )
            options_tokenized.append(option)

        # Storing Answer

        answer = data["label"]

        return {
            "article": article,
            "answer_index": answer_index,
            "options": options_tokenized,
            "answer": answer,
        }

    def __getitem__(self, idx):
        """
            Gets processed text, masks, options and answers at a particular index
            Args:
                idx (int): The index of the record to be fetched.
            Returns:
                sample(List) -> article,article_masks,options,answer_index,answer
            (all with tokenized input ids)
        """

        sample = []
        if self.config.preprocessed == False:
            datapoint = self._preprocess(self.data[idx])
        else:
            datapoint = self.data[idx]
        # Appending Article Input Ids
        sample.append(datapoint["article"]["input_ids"])
        # Appending Article Masks
        sample.append([1] * (len(datapoint["article"]["input_ids"])))

        options = []
        for i in range(5):
            options.append(datapoint["options"][i]["input_ids"])
        # Appending options input ids *WITH* CLS and SEP
        sample.append(options)
        # Appending Placeholder's index from the article
        sample.append(datapoint["answer_index"])
        # Appending correct answer choice
        sample.append(datapoint["answer"])

        return sample

    def custom_collate_fn(self, batch):
        # Used for batching; Pads all articles and options to equal token lengths
        articles = []
        article_masks = []
        options = []
        answer_indices = []
        answers = []

        max_len = 0
        ops_max_len = 0

        for sample in batch:

            articles.append(sample[0])
            max_len = max(max_len, len(sample[0]))
            article_masks.append(sample[1])

            options.append(sample[2])
            ops_max_len = max(ops_max_len, max([len(i) for i in sample[2]]))

            answer_indices.append(sample[3])
            answers.append(sample[4])

        # Applying padding to articles relative to the batch
        for i in range(len(articles)):

            articles[i] = articles[i] + [self.pad_id] * (max_len - len(articles[i]))
            article_masks[i] = article_masks[i] + [self.pad_id] * (
                max_len - len(article_masks[i])
            )

        # Applying padding to options relative to the batch
        for i in range(len(options)):
            for j in range(len(options[i])):
                options[i][j] = options[i][j] + [self.pad_id] * (
                    ops_max_len - len(options[i][j])
                )

        batch_sample = (
            articles,
            article_masks,
            options,
            answer_indices,
            answers,
        )
        return batch_sample
