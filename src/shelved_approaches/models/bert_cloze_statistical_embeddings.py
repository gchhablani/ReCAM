"""Implements Bert Cloze Style Question Answering"""
import torch
import math
import torch.nn as nn
from src.utils.mapper import configmapper
import numpy as np
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer

import glob
import nltk
from tqdm.auto import tqdm
from numpy.linalg import norm


nltk.download("punkt")

import spacy

nlp = spacy.load("en", disable=["parser", "ner"])

nltk.download("sentiwordnet")
from nltk.corpus import sentiwordnet as swn


from itertools import chain

nltk.download("wordnet")
from nltk.corpus import wordnet as wn


class StatisticalEmbedding:
    def __init__(self, normalise=True):
        # add word frequency later
        # try to fix number of senses and add it later
        # try to fix number of hyponyms and add it later
        self.normalise = normalise

    def get_embedding(self, word):
        len_embedding = self.get_length_of_word(word)
        depth_hypernymy_embedding = self.get_depth_of_hypernymy_tree(word)
        avg_depth_hypernymy_embedding = self.get_avg_depth_of_hypernymy_tree(word)
        pos_neg_obj_score = self.get_pos_neg_obj_scores(word)
        avg_pos_neg_obj_score = self.get_avg_pos_neg_obj_scores(word)

        embedding = [
            len_embedding,
            depth_hypernymy_embedding,
            avg_depth_hypernymy_embedding,
            pos_neg_obj_score[0],
            pos_neg_obj_score[1],
            pos_neg_obj_score[2],
            avg_pos_neg_obj_score[0],
            avg_pos_neg_obj_score[1],
            avg_pos_neg_obj_score[2],
        ]
        if self.normalise:
            embedding = embedding / norm(embedding)
        return embedding

    def get_length_of_word(self, word):
        words = word.split(" ")
        lengths = [len(word) for word in words]
        max_len = max(lengths)
        return max_len

    def get_depth_of_hypernymy_tree(self, word):
        max_len_paths = 0
        words = word.split(" ")
        for word_n in words:
            if len(wn.synsets(word_n)) > 0:
                j = wn.synsets(word_n)[0]
                paths_to_top = j.hypernym_paths()
                max_len_paths = max(
                    max_len_paths, len(max(paths_to_top, key=lambda i: len(i)))
                )

        return max_len_paths

    def get_avg_depth_of_hypernymy_tree(self, word):
        words = word.split(" ")
        lst_avg_len_paths = []
        for word_n in words:
            i = 0
            avg_len_paths = 0

            for j in wn.synsets(word_n):
                paths_to_top = j.hypernym_paths()
                max_len_path = len(max(paths_to_top, key=lambda k: len(k)))
                avg_len_paths += max_len_path
                i += 1
            if i > 0:
                return avg_len_paths / i
            else:
                return 0

    def get_pos_neg_obj_scores(self, word):
        words = word.split(" ")
        pos_scores = []
        neg_scores = []
        obj_scores = []

        for word_n in words:

            if len(list(swn.senti_synsets(word_n))) > 0:
                j = list(swn.senti_synsets(word_n))[0]

                pos_scores.append(j.pos_score())
                neg_scores.append(j.neg_score())
                obj_scores.append(j.obj_score())
            else:
                pos_scores.append(0)
                neg_scores.append(0)
                obj_scores.append(0)
        return (max(pos_scores), max(neg_scores), max(obj_scores))

    def get_avg_pos_neg_obj_scores(self, word):
        words = word.split(" ")
        pos_scores = []
        neg_scores = []
        obj_scores = []

        for word_n in words:
            ct = 0
            avg_pos_score = 0
            avg_neg_score = 0
            avg_obj_score = 0

            for j in list(swn.senti_synsets(word_n)):
                avg_pos_score += j.pos_score()
                avg_neg_score += j.neg_score()
                avg_obj_score += j.obj_score()
                ct += 1

            if ct > 0:
                pos_scores.append(avg_pos_score / ct)
                neg_scores.append(avg_neg_score / ct)
                obj_scores.append(avg_obj_score / ct)
            else:
                pos_scores.append(0)
                neg_scores.append(0)
                obj_scores.append(0)
        return (max(pos_scores), max(neg_scores), max(obj_scores))


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


@configmapper.map("models", "bert_cloze_ling_imp")
class BertForCloze(BertPreTrainedModel):
    """Implements Bert Cloze Model.
    Methods:
        forward(x_input): Returns the output of the Bert Cloze.
    """

    def __init__(self, config):

        super(BertForCloze, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights(self.cls)
        self.vocab_size = self.bert.embeddings.word_embeddings.weight.size(0)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.emb = StatisticalEmbedding()
        self.fc1 = nn.Linear(45, 5)
        self.fc2 = nn.Linear(10, 5)

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
        articles, articles_mask, ops, question_pos = x_input
        option_tokens = ops.clone()

        ### BERT CLOZE

        bsz = ops.size(0)
        ops = ops.reshape(bsz, 1, -1, 1)

        opnum = ops.size(1)
        out = self.bert(
            articles, attention_mask=articles_mask, output_hidden_states=False
        ).last_hidden_state

        question_pos = question_pos.reshape(-1, 1, 1)
        question_pos = question_pos.expand(bsz, opnum, out.size(-1))
        out = torch.gather(out, 1, question_pos)
        out = self.cls(out)

        # convert ops to one hot
        out = out.view(bsz, opnum, 1, self.vocab_size)
        out = out.expand(bsz, opnum, 5, self.vocab_size)
        out = torch.gather(out, 3, ops)

        out = out.view(-1, 5)

        ### Linguistic Features
        option_strings = []
        # convert the tokens to strings so that we can find the statistical embeddings
        for batch_of_options in range(bsz):
            batch_op = []
            for option_idx in range(option_tokens.shape[1]):
                opt_str = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        option_tokens[batch_of_options][option_idx].detach().tolist()
                    )
                )
                emb = self.emb.get_embedding(opt_str)
                batch_op.append(torch.tensor(emb))

            option_strings.append(torch.cat(batch_op, dim=0))

        option_strings = torch.as_tensor(
            torch.stack(option_strings), dtype=torch.float32, device=out.device
        )
        linguistic_output = self.fc1(option_strings)

        ### COMMON NETWORK
        bert_cat_linguistic = torch.cat((out, linguistic_output), dim=1)
        final_output = self.fc2(bert_cat_linguistic)

        return final_output
