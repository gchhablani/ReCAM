"""Implements Bert Cloze Style Question Answering with GSA layers on embeddings"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F


class GatedSelfAttention(nn.Module):
    def forward(self, Q, K, V):
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


@configmapper.map("models", "gsabert_cloze")
class GSABertForCloze(BertPreTrainedModel):
    """Implements GSABert Cloze Model.

    Methods:
        forward(x_input): Returns the output of the Bert Cloze.
    """

    def __init__(self, config):

        super(GSABertForCloze, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.gsalayers = [
            GatedSelfAttention() for i in range(2)
        ]  ## 2 layers of Gated Self Attention
        self.init_weights(self.cls)
        self.vocab_size = self.bert.embeddings.word_embeddings.weight.size(0)

    def init_weights(self, module):

        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x_input):
        """
        Return the output of Bert For Cloze.

        Args:
            x_input (torch.Tensor): Tensor List of articles, articles_mask, ops, question_pos

        Returns:
            x_output (torch.Tensor): The output regression scores for each option
        """
        articles, articles_mask, ops, question_pos = (
            x_input["articles"],
            x_input["article_attention_masks"],
            x_input["options"],
            x_input["answer_indices"],
        )
        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, -1, 1)

        opnum = ops.size(1)
        out = self.bert(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state

        ##GatedSelfAttention
        for layer in self.gsalayers:
            out = layer(out, out, out)

        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        out = self.cls(out)

        # convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)
        out = out.expand(bsz, opnum, 5, self.vocab_size)
        out = torch.gather(out, 3, ops)

        out = out.view(-1, 5)

        return out
