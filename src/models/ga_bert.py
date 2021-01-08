"""Implement GABERT Model for Reading Comprehension."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from src.utils.mapper import configmapper


import math
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


def process_options(options_tensor,func):
    single_option_shape = list(options_tensor.shape)
    single_option_shape[1]=1
    return torch.cat([func(torch.gather(options_tensor,1,torch.ones(single_option_shape,dtype=torch.int64)*i)) for i in range(5)],dim=1)


##DERIVED FROM BASELINE
class GatedAttention(nn.Module):
    def forward(self, question_states, article_states):
        question_att = question_states.permute(0,2,1)
        att_matrix = torch.bmm(article_states,question_att)

        att_weights = F.softmax(att_matrix.view(-1,att_matrix.size(-1)),dim=1).view_as(att_matrix)
        question_rep = torch.bmm(att_weights, question_states)

        question_to_article = torch.mul(article_states, question_rep)

        return question_to_article ##Attention applied on articles

class GABertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-uncased').embeddings

    def forward(self, article_tokens, question_tokens, options_tokens):
        article_embeds = self.embeddings(article_tokens)
        question_embeds = self.embeddings(question_tokens)
        options_embeds = process_options(options_tokens,self.embeddings)

        return article_embeds, question_embeds, options_embeds


class GABertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = BertModel.from_pretrained('bert-base-uncased')
        self.m2 = BertModel.from_pretrained('bert-base-uncased')
        self.ga = GatedAttention()

    def forward(self, article_contexts, question_contexts, article_attention_mask=None, question_attention_mask=None):

        for i in range(len(self.m1.encoder.layer)):
            current_layer_1 = self.m1.encoder.layer[i]
            current_layer_2 = self.m2.encoder.layer[i]
            question_contexts = current_layer_1(question_contexts, question_attention_mask.unsqueeze(1).unsqueeze(3))[0]
#             print(question_contexts.shape)
            article_intermediates = current_layer_2(article_contexts, article_attention_mask.unsqueeze(1).unsqueeze(3))[0]
#             print(article_intermediates.shape)
            article_contexts = self.ga(question_contexts,article_intermediates)

        return article_contexts, question_contexts

class GABertPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooler = BertModel.from_pretrained('bert-base-uncased').pooler
    def forward(self,contexts):
        return self.pooler(contexts)

##FROM BASELINE CODE
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        torch.nn.init.kaiming_normal_(self.linear.weight.data)
        torch.nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):

        # x: [batch_size, seq_len, in_features]
        x = self.linear(x)
        # x: [batch_size, seq_len, out_features]
        return x
##FROM BASELINE CODE
class MLPAttention(nn.Module):
    def __init__(self, dim, dropout):
        super(MLPAttention, self).__init__()

        self.Q_W = Linear(dim, dim)
        self.K_W = Linear(dim, dim)
        self.V_W = Linear(dim, dim)

        self.tanh = torch.nn.Tanh()
        self.V = Linear(dim, 1)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]
        # V: [batch_size, seq_len, dim]

#         print(Q)
#         print(K)
#         print(V)

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

##DERVIED FROM BASELINE
class BaselineOut(nn.Module):
    def __init__(self,dropout, hidden_size, output_dim):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mlp_att = MLPAttention(hidden_size, dropout)
        self.dot_layer = MLPAttention(hidden_size,dropout)
        self.final_linear = Linear(hidden_size*5,output_dim)
    def forward(self, article_contexts,question_contexts,options_embeds, answer_indices):

        single_question_context_shape = list(question_contexts.shape)
        single_question_context_shape[1] = 1


        ## Get the context for answer indices
        ## CAN ALSO GET JUST THE FIRST OUTPUT
        overall_question_context = torch.gather(question_contexts,1,torch.ones(single_question_context_shape,dtype=torch.int64)*answer_indices.reshape(-1,1,1)).squeeze(1)
        article_question_attention = self.mlp_att(overall_question_context, article_contexts, article_contexts)

#         print(article_question_attention.shape)

        options_attentions = process_options(options_embeds,lambda x: self.dropout(self.dot_layer(article_question_attention,x.squeeze(1),x.squeeze(1))))


        logits = self.dropout(self.final_linear(options_attentions))

        return logits

##BASED ON BERT-CLOZE
class GABertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
##BASED ON BERT-CLOZE
class GABertLMPredictionHead(nn.Module):
    def __init__(self, config, model_embedding_weights):
        super().__init__()
        self.transform = GABertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(
            model_embedding_weights.size(1),
            model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = model_embedding_weights
        self.bias = torch.nn.Parameter(
            torch.zeros(model_embedding_weights.size(0))
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)

        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
##BASED ON BERT-CLOZE
class ClozeStyleOut(nn.Module):
    def __init__(self,embeddings):
        super().__init__()
        bc = BertConfig.from_pretrained('bert-base-uncased')
        self.cls = GABertLMPredictionHead(bc,embeddings.word_embeddings.weight)
        self.vocab_size = embeddings.word_embeddings.weight.size(0)
        self.pad_token_id = bc.pad_token_id
    def forward(self, question_contexts,options_tokens, answer_indices):

        bsz = options_tokens.size(0)
        options_tokens = options_tokens.reshape(bsz,1,5,-1)
        opnum = options_tokens.size(1)


        ### CAN ALSO REPLACE WITH CODE IN LONGFORMERS_CLOZE
        single_question_context_shape = list(question_contexts.shape)
        single_question_context_shape[1] = 1
        out = torch.gather(question_contexts,1,torch.ones(single_question_context_shape,dtype=torch.int64)*answer_indices.reshape(-1,1,1))

        out = self.cls(out)

        out = out.view(bsz,opnum,1,self.vocab_size)
        out[:, :, :, self.pad_token_id] = 0
        out = out.expand(bsz, opnum, 5, self.vocab_size)

        out_tokens = torch.zeros((bsz, opnum, 5, 1), device=options_tokens.device)
        pad_tokens = options_tokens.shape[3] - torch.sum((options_tokens == self.pad_token_id), dim=3).unsqueeze(3)

        for i in range(options_tokens.shape[3]):
            ops_token = options_tokens[:, :, :, i].unsqueeze(3)
            out_tokens += torch.gather(out, 3, ops_token)

        out_tokens = torch.div(out_tokens, pad_tokens)
        out = out_tokens
        out = out.view(-1, 5)

        return out

## SIMPLE EMBEDDING GENERATOR
class GABert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = GABertEmbeddings()
        self.encoder = GABertEncoder()
    def forward(self,article_tokens, question_tokens, options_tokens, article_attention_masks=None, question_attention_masks = None):
        article_embeds, question_embeds, options_embeds = self.embeddings(article_tokens,question_tokens, options_tokens)
        article_contexts, question_contexts= self.encoder(article_embeds, question_embeds, article_attention_masks, question_attention_masks)

        return article_contexts, question_contexts, options_embeds

##UNUSED MODEL BASED ON BertModel
class GABertModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gabert = GABert()

        self.question_pooler = GABertPooler()
        self.article_pooler = GABertPooler()

    def forward(self,article_tokens, question_tokens, options_tokens, article_attention_masks=None, question_attention_masks = None):
        article_contexts, question_contexts, options_embeds = self.gabert(article_tokens, question_tokens,options_tokens, article_attention_masks, question_attention_masks)

        #all_contexts = torch.cat([article_contexts,question_contexts],dim=1)

        #return self.pooler(all_contexts)
        return article_contexts, question_contexts, options_embeds, self.article_pooler(article_contexts), self.question_pooler(question_contexts)

## JUST THE CLOZE STYLE HEAD
@configmapper.map("models","gabertcloze")
class GABertCloze(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model = GABert()
        self.cloze_out = ClozeStyleOut(self.model.embeddings.embeddings)
    def forward(self, batch):
        article_contexts, question_contexts,options_embeds = self.model(batch['articles_token_ids'], batch['questions_token_ids'], batch['options_token_ids'], batch['articles_attention_mask'], batch['questions_attention_mask'])
        cloze_logits = self.cloze_out(question_contexts,batch['options_token_ids'],batch['answer_indices'])
        return cloze_logits

## JUST THE GAREADER HEAD
@configmapper.map("models","gabertqa")
class GABertQA(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model = GABert()
        self.qa_out = BaselineOut(config.dropout, config.hidden_size,config.output_dim)
    def forward(self, batch):
        article_contexts, question_contexts,options_embeds = self.model(batch['articles_token_ids'], batch['questions_token_ids'], batch['options_token_ids'], batch['articles_attention_mask'], batch['questions_attention_mask'])
        qa_logits =  self.qa_out(article_contexts,question_contexts,options_embeds,batch['answer_indices'])
        return qa_logits

## COMBINE GAREADER HEAD AND CLOZESTYLE HEAD AND "POOL"
@configmapper.map("models","gabertclozeqa")
class GABertClozeAndQA(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model = GABert()
        self.qa_out = BaselineOut(config.dropout, config.hidden_size,config.output_dim)
        self.cloze_out = ClozeStyleOut(self.model.embeddings.embeddings)
        self.linear = torch.nn.Linear(config.output_dim*2,config.output_dim)
    def forward(self, batch):
        article_contexts, question_contexts,options_embeds = self.model(batch['articles_token_ids'], batch['questions_token_ids'], batch['options_token_ids'], batch['articles_attention_mask'], batch['questions_attention_mask'])

        qa_logits =  self.qa_out(article_contexts,question_contexts,options_embeds,batch['answer_indices'])
        cloze_logits = self.cloze_out(question_contexts,batch['options_token_ids'],batch['answer_indices'])

        concat = torch.cat([qa_logits,cloze_logits],dim=1)
        return self.linear(concat)
