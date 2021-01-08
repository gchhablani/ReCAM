"""Implements Roberta Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import RobertaModel, PreTrainedModel, RobertaConfig


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(RobertaLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(RobertaPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.LayerNorm = RobertaLayerNorm(config)

    def forward(self, hidden_states):
        # print(hidden_states)
        hidden_states = self.dense(hidden_states)
        # print(hidden_states)
        # exit()
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config, roberta_model_embedding_weights):
        super(RobertaLMPredictionHead, self).__init__()
        self.transform = RobertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            roberta_model_embedding_weights.size(1),
            roberta_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = roberta_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(roberta_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)

        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class RobertaOnlyMLMHead(nn.Module):
    def __init__(self, config, roberta_model_embedding_weights):
        super(RobertaOnlyMLMHead, self).__init__()
        self.predictions = RobertaLMPredictionHead(
            config, roberta_model_embedding_weights
        )

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@configmapper.map("models", "roberta_cloze")
class RobertaForCloze(RobertaPreTrainedModel):
    """Implements Roberta Cloze Model.

    Methods:
        forward(x_input): Returns the output of the Roberta Cloze.
    """

    def __init__(self, config):

        super(RobertaForCloze, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaOnlyMLMHead(
            config, self.roberta.embeddings.word_embeddings.weight
        )
        self.config = config

        self.init_weights(self.cls)
        self.vocab_size = self.roberta.embeddings.word_embeddings.weight.size(0)
        # print("MODEL EXPECT SIZE",self.vocab_size)

    def init_weights(self, module):

        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RobertaLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x_input):
        """
        Return the output of Roberta For Cloze.

        Args:
            x_input (torch.Tensor): Tensor List of articles, articles_mask, ops, question_pos

        Returns:
            x_output (torch.Tensor): The output regression scores for each option
        """
        pad_token_id = self.config.pad_token_id
        articles, articles_mask, ops, question_pos = x_input
        # print(ops.device)
        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, 5, -1)

        opnum = ops.size(1)
        out = self.roberta(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state

        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
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
