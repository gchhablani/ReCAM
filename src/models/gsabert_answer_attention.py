"""Implement GSAMN + BERT Model."""
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
        # self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]

        Q = Q.unsqueeze(1)  # [batch_size, 1, dim]

        scores = torch.matmul(K, Q.transpose(-1, -2))  # [batch_size, seq_len, 1]

        return scores


class GatedSelfAttention(nn.Module):
    def forward(Q, K, V):
        ## Q: [batch_size,seq_length,dim]
        ## K: [batch_size,seq_length,dim]
        ## V: [batch_size,seq_length,dim]
        K_att = K.permute(0, 2, 1)  ## [batch_size,dim,seq_length]

        attn_matrix = torch.bmm(Q, K_att)  ## [batch_size,seq_length, seq_length]

        attn_weights = F.softmax(
            attn_matrix.view(-1, attn_matrix.size(-1)), dim=1
        ).view_as(
            attn_matrix
        )  ## [batch_size,seq_length, seq_length]

        gated_attention_weights = torch.bmm(
            attn_weights, V
        )  ## [batch_size,seq_length,dim]

        gated_self_attention_update = torch.mul(gated_attention_weights, V)

        ## We don't do aggregation as we intend to use the options and mask embeddings.

        return gated_self_attention_update

        ## [batch_size,seq_length,dim]


@configmapper.map("models", "gsabert")
class GSABert(nn.Module):
    def __init__(self, config):
        super(GSABert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_pretrained_name)
        self.gsabertlayers = [GatedSelfAttention() for i in config.layers]
        self.mlpatt = MLPAttentionLogits(config.hidden_size, config.dropout)

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

        for layer in self.gsabertlayers:
            concat_embeddings = layer(
                concat_embeddings, concat_embeddings, concat_embeddings
            )

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

        out_logits = self.mlpatt(
            mask_embedding.squeeze(1), ops_avg_embeddings, ops_avg_embeddings
        ).squeeze(
            2
        )  ##[batch_size,5]

        return out_logits
