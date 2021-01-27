import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from src.utils.mapper import configmapper


@configmapper.map("datasets", "maxcloze")
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

        with open(self.config.file_path) as f:
            self.data = [json.loads(datapoint) for datapoint in f.read().splitlines()]

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # Check if truncate is true but truncate length has not been mentioned
        try:
            if self.config.truncate:
                try:
                    # Check to see if truncate_length exists
                    if self.config.truncate_length:
                        pass
                except KeyError:
                    # Default value 512
                    self.config.truncate_length = 512
        except KeyError:
            # Make False if Truncate does not Exist
            self.config.truncate = False

    def __len__(self):
        return len(self.data)

    def _preprocess(self, data):
        """
        Helper Function To preprocess each datapoint before __get_item__ for jsonl files
        """

        question_tokens = self.tokenizer(
            data["question"].replace("@placeholder", self.tokenizer.mask_token),
            add_special_tokens=False,
        )["input_ids"]
        article_tokens = self.tokenizer(data["article"], add_special_tokens=False)[
            "input_ids"
        ]

        if len(article_tokens) + len(question_tokens) + 9 > 512:
            remaining_length = self.config.truncate_length - len(question_tokens) - 9

            mask = [0 for i in range(len(article_tokens))]

            for i in range(len(article_tokens)):
                if article_tokens[i] in question_tokens:
                    mask[i] += 1

            max_context_index = 0
            min_context = -np.inf
            for i in range(len(article_tokens)):
                curr_context = min(np.sum(mask[:i]), np.sum(mask[i:]))
                if curr_context > min_context:
                    min_context = curr_context
                    max_context_index = i
            truncated_tokens = (
                [self.tokenizer.cls_token_id]
                + article_tokens[
                    max_context_index
                    - remaining_length // 2 : max_context_index
                    + remaining_length // 2
                ]
                + [self.tokenizer.sep_token_id]
                + question_tokens
                + [self.tokenizer.sep_token_id]
            )
        else:
            truncated_tokens = (
                [self.tokenizer.cls_token_id]
                + article_tokens
                + [self.tokenizer.sep_token_id]
                + question_tokens
                + [self.tokenizer.sep_token_id]
            )

        # Saving the [MASK]'s index
        answer_index = truncated_tokens.index(self.mask_id)

        # Tokenizing Options
        options = [data["option_" + str(i)] for i in range(5)]
        options_tokenized = []
        options_tokens = []
        for i in range(5):
            option = self.tokenizer(
                options[i],
                return_token_type_ids=False,
                return_attention_mask=False,
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
            options_tokenized.append(option)
            options_tokens += option

        # Storing Answer

        answer = data["label"]

        return {
            "article": truncated_tokens
            + options_tokens
            + [
                self.tokenizer.sep_token_id,
            ],
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
        datapoint = self._preprocess(self.data[idx])

        # Appending Article Input Ids
        sample.append(datapoint["article"])
        # Appending Article Masks
        sample.append([1] * (len(datapoint["article"])))

        options = []
        for i in range(5):
            options.append(datapoint["options"][i])
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
