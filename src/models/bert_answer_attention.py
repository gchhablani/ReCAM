"""Implement Answer Attention BERT Model."""
"""Implements Bert Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import BertConfig, BertModel, BertPreTrainedModel


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.LayerNorm = BertLayerNorm(config)

    def forward(self, hidden_states):
        # print(hidden_states)
        hidden_states = self.dense(hidden_states)
        # print(hidden_states)
        # exit()
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)

        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@configmapper.map("models", "answerbert")
class AnswerAttentionBert(nn.Module):
    def __init__(self, config):
        super(AnswerAttentionBert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_pretrained_name)
        bert_config = BertConfig.from_pretrained(self.config.bert_pretrained_name)
        self.cls = BertOnlyMLMHead(
            bert_config, self.bert.embeddings.word_embeddings.weight
        )
        self.vocab_size = self.bert.embeddings.word_embeddings.weight.size(0)

    def forward(self, batch):
        concats_token_ids = batch["concats_token_ids"]  # [batch_size,seq_length]
        concats_token_type_ids = batch[
            "concats_token_type_ids"
        ]  # [batch_size, seq_length]
        answer_indices = batch["answer_indices"]  # [batch_size,]
        concats_attention_masks = batch[
            "concats_attention_masks"
        ]  # [batch_size, seq_length]
        options = batch["options"]
        # ]  # [batch_size, 5, max_options_length]

        concat_embeddings = self.bert(
            input_ids=concats_token_ids,
            attention_mask=concats_attention_masks,
            token_type_ids=concats_token_type_ids,
        )[
            0
        ]  # [batch_size, seq_length, hidden_size]

        batch_size = answer_indices.shape[0]
        hidden_size = concat_embeddings.shape[-1]
        pad_token_id = 0

        options = options.reshape(batch_size, 1, 5, -1)

        opnum = options.size(1)
        out = self.bert(
            concats_token_ids,
            attention_mask=concats_attention_masks,
            token_type_ids=concats_token_type_ids,
            output_hidden_states=False,
        ).last_hidden_state

        answer_indices = answer_indices.reshape(-1, 1, 1)
        answer_indices = answer_indices.expand(batch_size, opnum, out.size(-1))
        out = torch.gather(out, 1, answer_indices)
        out = self.cls(out)
        # print(out.shape)
        # convert ops to one hot
        out = out.view(batch_size, opnum, 1, self.vocab_size)
        out[:, :, :, pad_token_id] = 0
        out = out.expand(batch_size, opnum, 5, self.vocab_size)
        # print(out.shape)
        out_tokens = torch.zeros((batch_size, opnum, 5, 1), device=options.device)
        pad_tokens = options.shape[3] - torch.sum(
            (options == pad_token_id), dim=3
        ).unsqueeze(3)

        for i in range(options.shape[3]):
            ops_token = options[:, :, :, i].unsqueeze(3)
            out_tokens += torch.gather(out, 3, ops_token)

        out_tokens = torch.div(out_tokens, pad_tokens)
        out = out_tokens
        out = out.view(-1, 5)

        return out
