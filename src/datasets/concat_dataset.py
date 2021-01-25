"""Implement ReCAM Dataset."""
import jsonlines
import torch
import math
from torch.utils.data import Dataset
from src.utils.mapper import configmapper


@configmapper.map("datasets", "concat")
class ConcatDataset(Dataset):
    def __init__(self, config, tokenizer):
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
        article = data["article"].lower()
        question = data["question"].lower().replace("@placeholder", self.mask_token)
        options = [data[f"option_{i}"] for i in range(5)]

        ques_ans = question + self.sep_token + " ".join([options[i] for i in range(5)])

        concat_tokenized = self.tokenizer(
            article,
            ques_ans,
            return_token_type_ids=True,
        )

        ## Need answer_index in the center if max_length exceeds the total length
        truncated_concat_token_ids = concat_tokenized["input_ids"][
            -self.config.truncate_length :
        ]
        truncated_concat_token_type_ids = concat_tokenized["token_type_ids"][
            -self.config.truncate_length :
        ]

        answer_index = truncated_concat_token_ids.index(self.mask_token_id)

        # sep_ids = [
        #     i
        #     for i, value in enumerate(truncated_concat_token_ids)
        #     if value == self.sep_token_id
        # ]
        # options_sep_ids = sep_ids[-2:]

        # options_indices = [
        #     list(range(options_sep_ids[i] + 1, options_sep_ids[i + 1]))
        #     for i in range(5)
        # ]

        options_tokenized = []
        for i in range(5):
            option = self.tokenizer(
                options[i],
                return_token_type_ids=False,
                return_attention_mask=False,
                add_special_tokens=False,
                verbose=False,
            )
            options_tokenized.append(option)

        return_dic = {
            "concat_token_ids": truncated_concat_token_ids,
            "concat_token_type_ids": truncated_concat_token_type_ids,
            "answer_index": answer_index,
            "options": options_tokenized,
            # "options_indices": options_indices,
        }
        if self.config.split == "test":
            return return_dic

        label = data["label"]

        return return_dic, label

    def __getitem__(self, idx):
        return_dic, label = self._preprocess(self.data[idx])
        return_dic["concat_attention_mask"] = [1] * len(return_dic["concat_token_ids"])
        # options_attention_masks = []
        # for i in return_dic["options_indices"]:
        #     options_attention_masks.append(len(i) * [1])
        # return_dic["options_attention_masks"] = options_attention_masks

        return return_dic, label

    def custom_collate_fn(self, batch):
        max_concat_len = 0
        max_options_len = 0

        concats = []
        concat_masks = []
        concat_token_type_ids = []
        answer_indices = []
        options = []
        # options_attention_masks = []
        labels = []

        for sample, label in batch:
            max_concat_len = max(max_concat_len, len(sample["concat_token_ids"]))
            concats.append(sample["concat_token_ids"])
            concat_masks.append(sample["concat_attention_mask"])
            concat_token_type_ids.append(sample["concat_token_type_ids"])
            answer_indices.append(sample["answer_index"])
            options.append(sample["options"])
            # options_attention_masks.append(sample["options_attention_masks"])
            for option in sample["options"]:
                max_options_len = max(max_options_len, len(option))

            if self.config.split != "test":
                labels.append(label)

        for i in range(len(concats)):
            concats[i] = concats[i] + [self.pad_token_id] * (
                max_concat_len - len(concats[i])
            )
            concat_masks[i] = concat_masks[i] + [0] * (
                max_concat_len - len(concat_masks[i])
            )

            concat_token_type_ids[i] = concat_token_type_ids[i] + [0] * (
                max_concat_len - len(concat_token_type_ids[i])
            )
            for j in range(len(options[i])):
                options[i][j] = options[i][j] + [self.pad_id] * (
                    max_options_len - len(options[i][j])
                )
            ##Setting Pad Token Type Id To 0

        # for idx in range(len(options_indices)):
        #     for idx2 in range(len(options_indices[idx])):
        #         options_indices[idx][idx2] = options_indices[idx][idx2] + [0] * (
        #             max_options_len - len(options_indices[idx][idx2])
        #         )  ##0th in the sequence would be CLS, since we use 'options_attention_masks' then values are zeroed out.
        #         options_attention_masks[idx][idx2] = options_attention_masks[idx][
        #             idx2
        #         ] + [0] * (
        #             max_options_len
        #             - len(
        #                 options_attention_masks[idx][idx2]
        #             )  ## 1 Million means not a valid index
        #         )

        return_dic = {
            "concats_token_ids": torch.LongTensor(concats),
            "concats_token_type_ids": torch.LongTensor(concat_token_type_ids),
            "answer_indices": torch.LongTensor(answer_indices),
            "concats_attention_masks": torch.FloatTensor(concat_masks),
            "options": torch.LongTensor(options),
            # "options_attention_masks": torch.LongTensor(options_attention_masks),
        }

        if self.config.split == "test":
            return return_dic

        # return_dic['labels']=torch.LongTensor(labels)
        return return_dic, torch.LongTensor(labels)
