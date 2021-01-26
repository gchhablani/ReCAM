"""Implements Bert Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import BertModel, BertPreTrainedModel


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
        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.transform_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.LayerNorm = BertLayerNorm(config)

    def forward(self, hidden_states):
        # print(hidden_states)
        hidden_states = self.dense(hidden_states)
        # hidden_states =  self.dense2(hidden_states)
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


@configmapper.map("models", "bert_cloze")
class BertForCloze(BertPreTrainedModel):
    """Implements Bert Cloze Model.

    Methods:
        forward(x_input): Returns the output of the Bert Cloze.
    """

    def __init__(self, config):

        super(BertForCloze, self).__init__(config)
        self.bert = BertModel(config)
        # freeze_layers = [i for i in range(22)]
        # self.bert = self.freeze_bert_fn(self.bert,freeze_layers)
        # self.dropout_layer = nn.Dropout(0.3)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights(self.cls)
        self.vocab_size = self.bert.embeddings.word_embeddings.weight.size(0)

    def freeze_bert_fn(
        self,
        model,
        freeze_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        freeze_embeddings=True,
    ):

        """
        Returns the model with the top "freeze_layers" layers frozen.

        Args:
            model (transformers.models.bert.modeling_bert.BertForSequenceClassification): model whose layers are to be frozen
            freeze_layers (list): The layers to be frozen
            freeze_embeddings (boolean)

        Returns:
            model (transformers.models.bert.modeling_bert.BertForSequenceClassification) with frozen layers
        """

        if freeze_embeddings:
            for param in list(model.embeddings.word_embeddings.parameters()):
                param.requires_grad = False
            print("Froze Embedding Layer")

        if len(freeze_layers) != 0:
            layer_indices = freeze_layers
            for layer_idx in layer_indices:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                print("Froze Layer: ", layer_idx)

        return model

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


        # articles, articles_mask, ops, question_pos = x_input
        # bsz = ops.size(0)
        # ops = ops.reshape(bsz, 1, -1, 1)

        # opnum = ops.size(1)
        # out = self.bert(
        #     articles, attention_mask=articles_mask, output_hidden_states=False
        # ).last_hidden_state

        # question_pos = question_pos.reshape(-1, 1, 1)
        # question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        # out = torch.gather(out, 1, question_pos)
        # out = self.dropout_layer(out)
        # out = self.cls(out)

        # # convert ops to one hot
        # out = out.view(bsz, opnum, 1, self.vocab_size)
        # out = out.expand(bsz, opnum, 5, self.vocab_size)
        # out = torch.gather(out, 3, ops)

        # out = out.view(-1, 5)

        # return out

        pad_token_id = self.config.pad_token_id
        articles, articles_mask, ops, question_pos = x_input
        # print(ops.device)
        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, 5, -1)

        opnum = ops.size(1)
        out = self.bert(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state

        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)

        # out = self.dropout_layer(out)
        out = self.cls(out)
        # print(out.shape)
        # convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)
        out[:, :, :, pad_token_id] = 0
        out = out.expand(bsz, opnum, 5, self.vocab_size)
        # print(out.shape)
        out_tokens = torch.zeros((bsz, opnum, 5, 1), device=ops.device)
        pad_tokens = ops.shape[3] - torch.sum((ops == pad_token_id), dim=3).unsqueeze(3)

        for i in range(ops.shape[3]):
            ops_token = ops[:, :, :, i].unsqueeze(3)
            out_tokens += torch.gather(out, 3, ops_token)

        out_tokens = torch.div(out_tokens, pad_tokens)
        out = out_tokens
        out = out.view(-1, 5)
        # print(out.shape)
        return out
