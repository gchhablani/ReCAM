"""Implement GAReader with BERT Embeddings"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.mapper import configmapper

# from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers import BertModel


def process_options(options_tensor, func):
    single_option_shape = list(options_tensor.shape)
    single_option_shape[1] = 1
    return torch.cat(
        [
            func(
                torch.gather(
                    options_tensor,
                    1,
                    torch.ones(
                        single_option_shape,
                        dtype=torch.int64,
                        device=options_tensor.device,
                    )
                    * i,
                )
            )
            for i in range(5)
        ],
        dim=1,
    )


# def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
#     lengths = torch.as_tensor(lengths, dtype=torch.int64)
#     lengths = lengths.cpu()
#     if enforce_sorted:
#         sorted_indices = None
#     else:
#         lengths, sorted_indices = torch.sort(lengths, descending=True)
#         sorted_indices = sorted_indices.to(input.device)
#         batch_dim = 0 if batch_first else 1
#         input = input.index_select(batch_dim, sorted_indices)

#     data, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(
#         input, lengths, batch_first
#     )
#     return PackedSequence(data, batch_sizes, sorted_indices)


# class LSTM(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         batch_first=False,
#         num_layers=1,
#         bidirectional=False,
#         dropout=0.2,
#     ):
#         super(LSTM, self).__init__()

#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=bidirectional,
#             batch_first=batch_first,
#         )
#         self.reset_params()
#         self.dropout = nn.Dropout(p=dropout)

#     def reset_params(self):
#         for i in range(self.rnn.num_layers):
#             nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{i}"))
#             nn.init.kaiming_normal_(getattr(self.rnn, f"weight_ih_l{i}"))
#             nn.init.constant_(getattr(self.rnn, f"bias_hh_l{i}"), val=0)
#             nn.init.constant_(getattr(self.rnn, f"bias_ih_l{i}"), val=0)
#             bias = getattr(self.rnn, f"bias_hh_l{i}").detach()
#             bias.chunk(4)[1].fill_(1)
#             with torch.no_grad():
#                 setattr(self.rnn, f"bias_hh_l{i}", nn.Parameter(bias))

#             if self.rnn.bidirectional:
#                 nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{i}_reverse"))
#                 nn.init.kaiming_normal_(getattr(self.rnn, f"weight_ih_l{i}_reverse"))
#                 nn.init.constant_(getattr(self.rnn, f"bias_hh_l{i}_reverse"), val=0)
#                 nn.init.constant_(getattr(self.rnn, f"bias_ih_l{i}_reverse"), val=0)
#                 bias = getattr(self.rnn, f"bias_hh_l{i}_reverse").detach()
#                 bias.chunk(4)[1].fill_(1)
#                 with torch.no_grad():
#                     setattr(self.rnn, f"bias_hh_l{i}_reverse", nn.Parameter(bias))

#     def forward(self, x, x_len):
#         # x: [batch_size, seq_len, dim], x_len:[batch_size]
#         x_len_sorted, x_idx = torch.sort(x_len, descending=True)
#         x_sorted = torch.index_select(x, dim=0, index=x_idx)
#         sorted_x, x_ori_idx = torch.sort(x_idx)

#         # x_packed = nn.utils.rnn.pack_padded_sequence(
#         #     x_sorted, x_len_sorted, batch_first=True
#         # )
#         x_packed = hotfix_pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
#         x_packed, (hidden, c) = self.rnn(x_packed)

#         x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
#         x = x.index_select(dim=0, index=x_ori_idx)

#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         # hidden = hidden.permute(1, 0, 2).contiguous().view(-1,
#         #                                          hidden.size(0) * hidden.size(2)).squeeze()
#         hidden = hidden.index_select(dim=0, index=x_ori_idx)

#         return hidden, x


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


class MLPAttention(nn.Module):
    def __init__(self, dim, dropout):
        super(MLPAttention, self).__init__()

        self.Q_W = Linear(dim, dim)
        self.K_W = Linear(dim, dim)
        self.V_W = Linear(dim, dim)

        self.tanh = nn.Tanh()
        self.V = Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]
        # V: [batch_size, seq_len, dim]

        Q = self.dropout(self.Q_W(Q))  # [batch_size, dim]
        K = self.dropout(self.K_W(K))  # [batch_size, seq_len, dim]
        V = self.dropout(self.V_W(V))  # [batch_size, seq_len, dim]

        Q = Q.unsqueeze(1)  # [batch_size, 1, dim]
        M = self.dropout(self.tanh(Q + K))  # [batch_size, seq_len, dim]
        scores = self.dropout(self.V(M))  # [batch_size, seq_len, 1]
        scores = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

        R = self.dropout(V * scores)  # [batch_size, seq_len, dim]

        feat = torch.sum(R, dim=1)  # [batch_size, dim]

        return feat


def gated_attention(article, question):
    """
    Args:
        article: [batch_size, article_len , dim]
        question: [batch_size, question_len, dim]
    Returns:
        question_to_article: [batch_size, article_len, dim]
    """
    question_att = question.permute(0, 2, 1)
    # question : [batch_size * dim * question_len]

    att_matrix = torch.bmm(article, question_att)
    # att_matrix: [batch_size * article_len * question_len]

    att_weights = F.softmax(att_matrix.view(-1, att_matrix.size(-1)), dim=1).view_as(
        att_matrix
    )
    # att_weights: [batch_size, article_len, question_len]

    question_rep = torch.bmm(att_weights, question)
    # question_rep : [batch_size, article_len, dim]

    question_to_article = torch.mul(article, question_rep)
    # question_to_article: [batch_size, article_len, dim]

    return question_to_article


@configmapper.map("models", "gareader_bert")
class GAReaderBert(nn.Module):
    """
    Baseline GAReader with BERT Embeddings
    """

    def __init__(
        self,
        config,
    ):
        super(GAReaderBert, self).__init__()
        self.config = config

        # self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.bert = BertModel.from_pretrained(config.pretrained_bert_name)

        self.rnn = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            config.rnn_num_layers,
            True,
            True,
            config.dropout,
            config.bidirectional,
        )

        self.ga_rnn = nn.LSTM(
            config.hidden_size * (2 if config.bidirectional else 1),
            config.hidden_size,
            config.rnn_num_layers,
            True,
            True,
            config.dropout,
            config.bidirectional,
        )

        self.ga_layers = config.ga_layers

        self.mlp_att = MLPAttention(config.hidden_size * 2, config.dropout)

        self.dot_layer = MLPAttention(config.hidden_size * 2, config.dropout)

        self.final_linear = Linear(config.hidden_size * 10, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, batch):

        article, question, options, answer_indices, article_masks, question_masks = (
            batch["articles_token_ids"],
            batch["questions_token_ids"],
            batch["options_token_ids"],  # [batch_size,5,seq_length]
            batch["answer_indices"],
            batch["articles_attention_mask"],
            batch["questions_attention_mask"],
        )

        article_embeds = self.dropout(self.bert(article)[0])
        # article_embeds: [batch_size, seq_len, hidden_size]

        question_embeds = self.dropout(self.bert(question)[0])
        # question_embeds: [batch_size, seq_len, hidden_size]

        # options_embeds: [batch_size,5,seq_len, hidden_size]

        question_out, question_hidden = self.rnn(question_embeds)
        # question_out: [batch_size, seq_len, hidden_size * 2]

        options_out = []

        for i in range(5):
            options_i = options[:, i, :]  ##batch_size,seq_length
            options_i_embeds = self.bert(options_i)
            options_i_out, _ = self.rnn(options_i_embeds)
            options_out.append(options_i_out)

        # option_out: list of [batch_size,seq_length,hidden_size*2] of length 5

        article_out, _ = self.rnn(article_embeds)
        # article_out: [batch_size, seq_length hidden_size * 2]

        for layer in range(self.ga_layers):

            article_emb = self.dropout(gated_attention(article_out, question_out))
            # article_emb: [batch_size, seq_length, hidden_size * 2]

            article_out, _ = self.ga_rnn(article_emb)
            # article_out: [batch_size, seq_length, hidden_size * 2]

        ATT_article_question = self.dropout(
            self.mlp_att(
                question_hidden[0]
                .view(
                    self.config.rnn_num_layers,
                    (2 if self.config.bidirectional else 1),
                    -1,
                    self.config.hidden_size,
                )[self.config.rnn_num_layers - 1 :, :, :, :]
                .view(
                    -1,
                    self.config.hidden_size * (2 if self.config.bidirectional else 1),
                ),
                article_out,
                article_out,
            )
        )
        # ATT_article_question: [batch_size, hidden_size * 2]

        # 融合 option 信息 [batch_size, hidden_size * 2]

        ATT_option = []
        for i in range(5):
            ATT_option_i = self.dropout(
                self.dot_layer(ATT_article_question, options_out[i], options_out[i])
            )
            ATT_option.append(ATT_option_I)

        all_infomation = torch.cat(ATT_option, dim=1)

        logit = self.dropout(self.final_linear(all_infomation))

        return logit
