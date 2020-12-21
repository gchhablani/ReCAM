import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json


class ReCAMDataset(Dataset):
    def __init__(self, data_path, device="cuda"):
        self.data = torch.load(data_path)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        returns: sample -> List ([article,article_masks,options,answer_index,answer])
        (all with tokenized input ids)
        """

        # self.data[idx]['article_mask'] = [1]*(len(self.data[idx]['article']['input_ids']))
        """
        for i in range(5):
            self.data[idx]['options'][i]['input_ids'] = torch.tensor(self.data[idx]['options'][i]['input_ids'],device=self.device)
        """
        # self.data[idx]['answer'] =  torch.tensor(self.data[idx]['answer'],device=self.device)

        sample = []

        # Appending Article Input Ids
        sample.append(self.data[idx]["article"]["input_ids"])
        # Appending Article Masks
        sample.append([1] * (len(self.data[idx]["article"]["input_ids"])))

        options = []
        for i in range(5):
            options.append(self.data[idx]["options"][i]["input_ids"])
        # Appending options input ids *WITH* CLS and SEP
        sample.append(options)
        # Appending Placeholder's index from the article
        sample.append(self.data[idx]["answer_index"])
        # Appending correct answer choice
        sample.append(self.data[idx]["answer"])

        return sample

    def custom_collate_fn(self, batch):
        articles = []
        article_masks = []
        options = []
        answer_indices = []
        answers = []

        max_len = 0
        ops_max_len = 0

        # print("BATCH:" + str(batch))

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

            articles[i] = articles[i] + [0] * (max_len - len(articles[i]))
            article_masks[i] = article_masks[i] + [0] * (
                max_len - len(article_masks[i])
            )

        # Applying padding to options relative to the batch
        for i in range(len(options)):
            for j in range(len(options[i])):
                options[i][j] = options[i][j] + [0] * (ops_max_len - len(options[i][j]))

        # Converting entire batch to tensors
        batch_sample = (
            torch.tensor(articles, device=self.device),
            torch.tensor(article_masks, device=self.device),
            torch.tensor(options, device=self.device),
            torch.tensor(answer_indices, device=self.device),
            torch.tensor(answers, device=self.device),
        )
        return batch_sample


def preprocessor(data_path, save_path, tokenizer, truncate=False, truncate_length=512):
    mask_id = tok.convert_tokens_to_ids("[MASK]")
    with open(data_path) as f:
        data_items = [json.loads(datapoint) for datapoint in f.read().splitlines()]
    print("Starting to Preprocess...")
    datapoints = []
    for data in data_items:
        # Tokenizing article+question , replacing the blanks with [MASK].
        article = (
            data["article"].lower()
            + " "
            + data["question"].lower().replace("@placeholder", "[MASK]")
        )
        article = tokenizer(
            article, return_token_type_ids=False, return_attention_mask=False
        )

        if truncate and len(article["input_ids"]) > truncate_length:
            article["input_ids"] = article["input_ids"][-truncate_length:]

        # Saving the [MASK]'s index
        answer_index = article["input_ids"].index(mask_id)

        # Tokenizing Options
        options = [data["option_" + str(i)] for i in range(5)]
        options_tokenized = []
        for i in range(5):
            option = tokenizer(
                options[i], return_token_type_ids=False, return_attention_mask=False
            )
            options_tokenized.append(option)

        # Storing Answer

        answer = data["label"]

        # Appending data

        datapoints.append(
            {
                "article": article,
                "answer_index": answer_index,
                "options": options_tokenized,
                "answer": answer,
            }
        )

    print("Preprocessing Completed!")
    # Save Data to a file
    torch.save(datapoints, save_path)


""" 
TEST:
from transformers import AutoTokenizer
from bert_cloze_dataloader import *
tok = AutoTokenizer.from_pretrained('bert-large-uncased',mask_token = '[MASK]')
#tok = AutoTokenizer.from_pretrained('allenai/longformer-base-4096',mask_token = '[MASK]')
Preprocessor('/content/SemEval2021-Reading-Comprehension-of-Abstract-Meaning/data/training_data/Task_1_dev.jsonl','/content/train.pt',tok)
re = ReCAMDataset('/content/train.pt')
dl = DataLoader(re,collate_fn = re.custom_collate_fn, batch_size = 4)
count = 0
for i in dl:
    count += 1
    #print(i)
    if count == 10:
        break



"""
