"""Implements DistilBert Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
from transformers import DistilBertModel, DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@configmapper.map("models", "distilbert_cloze")
class DistilBertForCloze(DistilBertPreTrainedModel):
    """Implements DistilBert Cloze Model.

    Methods:
        forward(x_input): Returns the output of the DistilBert Cloze.
    """

    def __init__(self, config):

        super(DistilBertForCloze, self).__init__(config)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.config = config
        self.dropout_layer = nn.Dropout(0.3)


        self.vocab_size = self.distilbert.embeddings.word_embeddings.weight.size(0)


    def forward(self, x_input):
        """
        Return the output of DistilBert For Cloze.

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
        out = self.distilbert(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state


        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)

        # Dropout
        out = self.dropout_layer(out)
        

        # out = self.cls(out)
        prediction_logits = self.vocab_transform(out)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        out = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        # convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)
        out[:, :, :, pad_token_id] = 0
        out = out.expand(bsz, opnum, 5, self.vocab_size)

        out_tokens = torch.zeros((bsz, opnum, 5, 1), device=ops.device)
        pad_tokens = ops.shape[3] - torch.sum((ops == pad_token_id), dim=3).unsqueeze(3)

        for i in range(ops.shape[3]):
            ops_token = ops[:, :, :, i].unsqueeze(3)
            out_tokens += torch.gather(out, 3, ops_token)

        out_tokens = torch.div(out_tokens, pad_tokens)
        out = out_tokens
        out = out.view(-1, 5)

        return out