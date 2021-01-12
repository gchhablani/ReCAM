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
        self.V_W = Linear(dim, dim)

        self.linear = Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]
        # V: [batch_size, seq_len, dim]

        Q = self.dropout(self.Q_W(Q))  # [batch_size, dim]
        K = self.dropout(self.K_W(K))  # [batch_size, seq_len, dim]
        V = self.dropout(self.V_W(V))  # [batch_size, seq_len, dim]

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
        options_indices = batch[
            "options_indices"
        ]  # [batch_size, 5, max_options_length] May be padded using 1000000
        options_attention_masks = batch[
            "options_attention_masks"
        ]  # [batch_size, 5, max_options_length]

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
        mask_embedding = torch.gather(
            concat_embeddings, 1, answer_indices
        )  # [batch_size,1,768]

        tokens_per_option_per_batch = torch.sum(
            options_attention_masks, dim=2
        )  ##[batch_size,5]

        ops_avg_embeddings = []

        for i in range(5):
            ops_i_indices = options_indices[:, i, :]  ## [batch_size,max_options_length]
            ops_i_indices = ops_i_indices.reshape(batch_size, -1, 1).expand(
                batch_size, -1, hidden_size
            )  ##[batch_size,max_options_length,hidden_size]
            ops_i_masks = (
                options_attention_masks[:, i, :]
                .reshape(batch_size, -1, 1)
                .expand(batch_size, -1, hidden_size)
            )  ##[batch_size,max_options_length,hidden_size]
            ops_i_embeddings = torch.gather(
                concat_embeddings, 1, ops_i_indices
            )  ## [batch_size,max_options_length,hidden_size]
            ops_i_embeddings = (
                ops_i_masks * ops_i_embeddings
            )  ## [batch_size,max_options_length,hidden_size]
            ops_i_avg_embeddings = torch.sum(
                ops_i_embeddings, dim=1
            ) / tokens_per_option_per_batch[:, i].reshape(-1, 1).expand(
                -1, hidden_size
            )  ##[batch_size,hidden_size]
            ops_avg_embeddings.append(ops_i_avg_embeddings.unsqueeze(1))
        ops_avg_embeddings = torch.cat(ops_avg_embeddings, dim=1)

        out_logits = self.attention(
            mask_embedding.squeeze(), ops_avg_embeddings, ops_avg_embeddings
        ).squeeze()

        return out_logits
