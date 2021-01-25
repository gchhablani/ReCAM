"""Implement Answer Attention BERT Model."""
from torch import nn
from transformers import BertPreTrainedModel, BertConfig, BertModel
from src.utils.mapper import configmapper
import torch
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight.data)
        nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):

        # x: [batch_size, seq_len, in_features]
        x = self.linear(x)
        # x: [batch_size, seq_len, out_features]
        return x


class MLPAttentionLogits(nn.Module):
    def __init__(self, dim, dropout):
        super(MLPAttentionLogits, self).__init__()

        self.Q_W = Linear(dim, dim)
        self.K_W = Linear(dim, dim)

        self.linear = Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]

        Q = self.dropout(self.Q_W(Q))  # [batch_size, dim]
        K = self.dropout(self.K_W(K))  # [batch_size, seq_len, dim]

        Q = Q.unsqueeze(1)  # [batch_size, 1, dim]

        M = self.dropout(Q * K)  # [batch_size, seq_len, dim]
        scores = self.dropout(self.linear(M))  # [batch_size, seq_len, 1]

        return scores


@configmapper.map("models", "answerbert")
class AnswerAttentionBert(nn.Module):
    def __init__(self, config):
        super(AnswerAttentionBert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_pretrained_name)
        self.attention = MLPAttentionLogits(
            self.config.hidden_size, self.config.dropout
        )

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
        answer_indices = answer_indices.reshape(-1, 1, 1).expand(
            batch_size, 1, hidden_size
        )  # [batch_size, 1, hidden_size]

        pad_token_id = self.config.pad_token_id

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
