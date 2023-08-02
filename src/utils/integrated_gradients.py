import gc
import os
import pickle as pkl

import numpy as np
from captum.attr import IntegratedGradients
from datasets import Dataset
import json
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import tensorflow as tf


class MyIntegratedGradients:
    def __init__(self, config, model, dataset, tokenizer):
        self.config = config

        self.model = model
        self.dataset = dataset
        self.model.eval()
        self.model.to(torch.device(self.config.device))
        self.model.zero_grad()
        self.tokenizer = tokenizer

        with open(self.dataset.config.file_path) as f:
            self.original_data = [
                json.loads(datapoint) for datapoint in f.read().splitlines()
            ]

    def get_embedding_outputs(self, input_ids):
        return self.model.bert.embeddings(input_ids)

    def get_model_outputs(self, hidden_states, attention_masks, ops, question_pos):
        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, 5, -1)

        opnum = ops.size(1)
        # print(hidden_states.shape,attention_masks.shape)
        extended_attention_masks = torch.tensor(
            self.get_extended_attention_mask(
                attention_masks, hidden_states.shape[0:2], "float32"
            )
            .cpu()
            .numpy()
        ).cuda()
        out = self.model.bert.encoder(
            hidden_states, extended_attention_masks, return_dict=None
        )[0]
        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        # out = self.dropout_layer(out)
        out = self.model.cls(out)
        # print(out.shape)
        # convert ops to one hot
        out = out.view(bsz, opnum, 1, self.model.vocab_size)
        out[:, :, :, self.model.config.pad_token_id] = 0
        out = out.expand(bsz, opnum, 5, self.model.vocab_size)
        # print(out.shape)
        out_tokens = torch.zeros((bsz, opnum, 5, 1), device=ops.device)
        pad_tokens = ops.shape[3] - torch.sum(
            (ops == self.model.config.pad_token_id), dim=3
        ).unsqueeze(3)

        for i in range(ops.shape[3]):
            ops_token = ops[:, :, :, i].unsqueeze(3)
            out_tokens += torch.gather(out, 3, ops_token)

        out_tokens = torch.div(out_tokens, pad_tokens)
        out = out_tokens
        out = out.view(-1, 5)
        # print(out.shape)
        return F.softmax(out, dim=1)

    def get_token_wise_attributions(
        self, embedding_outputs, attention_masks, ops, question_pos, max_logits
    ):
        int_grad = IntegratedGradients(
            self.get_model_outputs,
            multiply_by_inputs=True,
        )
        attributions, approximation_error = int_grad.attribute(
            embedding_outputs,
            target=max_logits,
            n_steps=self.config.n_steps,
            method=self.config.method,
            additional_forward_args=(attention_masks, ops, question_pos),
            internal_batch_size=self.config.internal_batch_size,
            return_convergence_delta=True,
        )

        return {
            "attributions": attributions,
            "delta": approximation_error,
        }

    def get_token_wise_importances(
        self, per_example_input_ids, per_example_attributions
    ):
        """Normalize the token wise attributions after taking a norm.

        Args:
            per_example_input_ids (torch.tensor): The input_ids for the examle.
            per_example_attributions (torch.tensor): The attributions for the tokens.

        Returns:
            list,np.ndarray: The tokens list and the numpy array of importances.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(per_example_input_ids)
        token_wise_attributions = torch.linalg.norm(per_example_attributions, dim=1)
        # [batch_size,seq_length] = Norm of [batch_size, seq_length, hidden_size]
        token_wise_importances = token_wise_attributions / torch.sum(
            token_wise_attributions, dim=0
        ).reshape(
            -1, 1
        )  # Normalize by sum across seq_length

        return (
            tokens,
            token_wise_importances.squeeze(0).detach().cpu().numpy(),
        )

    def get_word_wise_importances(
        self,
        per_example_input_ids,
        per_example_offset_mapping,
        per_example_token_wise_importances,
        per_example_text,
    ):
        tokens = self.tokenizer.convert_ids_to_tokens(per_example_input_ids)
        offset_mapping = per_example_offset_mapping
        word_wise_importances = []
        word_wise_offsets = []
        word_wise_category = []
        words = []
        is_context = False
        for i, token in enumerate(tokens):
            if token == "[SEP]":
                is_context = not is_context
                continue
            if token == "[CLS]":
                is_context = False
                continue

            if token == "[PAD]":
                continue

            if token.startswith("##"):
                if (
                    tokens[i - 1] == "[SEP]"
                ):  # Tokens can be broked due to stride after the [SEP]
                    word_wise_importances.append(
                        per_example_token_wise_importances[i]
                    )  # We just make new entries for them
                    word_wise_offsets.append(offset_mapping[i])

                    words.append(
                        per_example_text[
                            word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                        ]
                    )

                else:
                    word_wise_importances[-1] += per_example_token_wise_importances[i]
                    word_wise_offsets[-1] = (
                        word_wise_offsets[-1][0],
                        offset_mapping[i][1],
                    )
                    words[-1] = per_example_text[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]

            else:
                # print(offset_mapping,'\n',i)
                word_wise_importances.append(per_example_token_wise_importances[i])
                word_wise_offsets.append(offset_mapping[i])
                words.append(
                    per_example_text[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]
                )

        if (
            np.sum(word_wise_importances) == 0
            or np.sum(word_wise_importances) == np.nan
            or np.sum(word_wise_importances) == np.inf
        ):
            print(np.sum(word_wise_importances))
            print(words)
            print(tokens)
        return (
            words,
            word_wise_importances / np.sum(word_wise_importances),
            word_wise_category,
        )  ## Normalized Scores

    def get_importances(self, examples):

        overall_word_importances = []
        overall_token_importances = []

        dataloader = DataLoader(
            examples,
            collate_fn=examples.custom_collate_fn,
            batch_size=self.config.internal_batch_size,
            shuffle=False,
        )
        # for batch_idx in tqdm(range(0, len(examples), self.config.internal_batch_size)):
        idx_for_original_samples = 0
        for inputs in tqdm(dataloader):
            # batch = examples[batch_idx : batch_idx + self.config.internal_batch_size]
            columns = [
                "input_ids",
                "attention_mask",
                "ops",
                "question_pos",
                "offset_mapping",
            ]
            batch = {}
            for index in range(len(columns)):
                batch[columns[index]] = torch.tensor(
                    inputs[index], device=torch.device(self.config.device)
                )
            # print(batch["offset_mapping"])
            embedding_outputs = self.get_embedding_outputs(batch["input_ids"])
            # print(embedding_outputs.shape)
            logits = self.get_model_outputs(
                embedding_outputs,
                batch["attention_mask"],
                batch["ops"],
                batch["question_pos"],
            )
            max_logits = torch.argmax(logits, dim=1)

            attributions = self.get_token_wise_attributions(
                embedding_outputs,
                batch["attention_mask"],
                batch["ops"],
                batch["question_pos"],
                max_logits,
            )

            token_importances = []
            word_importances = []

            gc.collect()
            token_importances.append([])
            word_importances.append([])
            for (example_index, attributions,) in enumerate(
                attributions["attributions"]
            ):  # attribution_shape = [seq_length,hidden_size]
                input_ids = batch["input_ids"][example_index]
                token_wise_importances = self.get_token_wise_importances(
                    input_ids, attributions
                )
                token_importances[-1].append(token_wise_importances)

                text = (
                    self.original_data[idx_for_original_samples]["article"]
                    + " "
                    + self.original_data[idx_for_original_samples]["question"].replace(
                        "@placeholder", "[MASK]"
                    )
                )
                # print(text)
                offset_mapping = batch["offset_mapping"][example_index]
                word_wise_importances = self.get_word_wise_importances(
                    input_ids,
                    offset_mapping,
                    token_wise_importances[1],
                    text,
                )
                word_importances[-1].append(word_wise_importances)
                idx_for_original_samples += 1
            overall_word_importances.append(word_importances)
            overall_token_importances.append(token_importances)
        return {
            "word_importances": overall_word_importances,
            # batches, batch_size, len of examples
            "token_importances": overall_token_importances,
            # batches,len of layers, batch_size, len of examples
        }

    def rearrange_importances(self, importances):
        num_batches = len(importances)
        batch_size = len(importances[0][0])
        # num_batches, num_layers, num_samples, 2 -> num_layers, num_samples*num_batches, 2

        num_samples = 0
        num_samples = np.sum(len(batch) for batch in importances)
        sample_wise = [[] for _ in range(num_samples)]
        j = 0
        for batch_idx in range(num_batches):
            for sample_idx in range(
                len(importances[batch_idx])
            ):  ## Some batches might not be filled up to batch_size
                sample_wise[j].append(importances[batch_idx][sample_idx])
                j += 1

        return sample_wise

    def get_random_samples_and_importances(self, n_samples=1000):

        if n_samples > len(self.dataset):
            raise ValueError(
                "n_samples cannot be greater than the samples in validation_dataset"
            )
        np.random.seed(42)
        random_indices = list(
            np.random.choice(
                list(range(len(self.dataset["input_ids"]))),
                size=n_samples,
                replace=False,
            )
        )
        # print(random_indices)
        print("### Sampling ###")
        samples = Dataset.from_dict(self.dataset[random_indices])
        print("## Samples ##")
        print(samples)

        importances = self.get_importances(samples)
        word_importances = self.rearrange_importances(importances["word_importances"])
        token_importances = self.rearrange_importances(importances["token_importances"])

        # print(len(word_importances))
        # print(len(word_importances[0]))
        # print(len(word_importances[0][0]))
        return samples, word_importances, token_importances

    def get_all_importances(
        self,
    ):
        importances = self.get_importances(self.dataset)
        word_importances = self.rearrange_importances(importances["word_importances"])
        token_importances = self.rearrange_importances(importances["token_importances"])

        return self.dataset, word_importances, token_importances

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        attention_mask = attention_mask.cpu()
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        extended_attention_mask = tf.reshape(
            attention_mask, (input_shape[0], 1, 1, input_shape[1])
        )

        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
