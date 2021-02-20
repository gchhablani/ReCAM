"""Implements Roberta Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaLMHead,
)


@configmapper.map("models", "roberta_cloze")
class RobertaForCloze(RobertaPreTrainedModel):
    """Implements Roberta Cloze Model.

    Methods:
        forward(x_input): Returns the output of the Roberta Cloze.
    """

    def __init__(self, config):

        super(RobertaForCloze, self).__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.cls = RobertaLMHead(config)
        self.dropout_layer = nn.Dropout(0.3)

        self.init_weights()
        self.config = config
        self.vocab_size = self.roberta.embeddings.word_embeddings.weight.size(0)

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

        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, 5, -1)

        opnum = ops.size(1)
        out = self.roberta(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state

        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        out = self.dropout_layer(out)
        out = self.cls(out)

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
        return out
