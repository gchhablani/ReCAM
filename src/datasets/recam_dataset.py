"""Implement ReCAM Dataset"""
import jsonlines
import torch
from torch.utils.data import Dataset
from src.utils.mapper import configmapper

@configmapper.map("datasets","recam")
class ReCAMDataset(Dataset):
    def __init__(self,config,tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with jsonlines.open(self.config.file_path) as f:
            self.data = list(f)
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def __len__(self):
        return len(self.data)

    def _preprocess(self,data):
        article = data['article'].lower()
        question = data['question'].lower().replace('@placeholder',self.tokenizer.mask_token)


        article_token_ids = self.tokenizer.encode(article)
        question_token_ids = self.tokenizer.encode(question)
        answer_index = question_token_ids.index(self.mask_id)


        article_token_ids = article_token_ids[:self.config.article_truncate_length]
        beg = max(0,answer_index-(self.config.question_truncate_length/2))
        end = min(len(question_token_ids),answer_index+(self.config.question_truncate_length/2))

        ## Need answer_index in the center if max_length exceeds the total length
        question_token_ids = question_token_ids[beg:end]
        answer_index = question_token_ids.index(self.mask_id) ## Fix answer index again

        options_token_ids = [self.tokenizer.encode(data[f'option_{i}'].lower()) for i in range(5)]


        return_dic = {
            "article_token_ids":article_token_ids,
            "question_token_ids":question_token_ids,
            "options_token_ids":options_token_ids,
            "answer_index":answer_index,
        }
        if self.config.split=="test":
            return return_dic

        label = data['label']

        return return_dic, label

    def __getitem__(self,idx):
        return_dic,labels = self._preprocess(self.data[idx])
#         return_dic["article_attention_mask"] = [1]*len(return_dic["article_token_ids"])
#         return_dic["question_attention_mask"] = [1]*len(return_dic["question_token_ids"])
        return_dic["article_attention_mask"] = [1]*len(return_dic["article_token_ids"])
        return_dic["question_attention_mask"] = [1]*len(return_dic["question_token_ids"])

        return return_dic,labels

    def custom_collate_fn(self, batch):
        max_article_len = 0
        max_question_len = 0
        max_options_len = 0

        articles = []
        article_masks = []
        questions = []
        question_masks = []
        answer_indices = []
        options = []
        labels = []

        for sample,label in batch:
            max_article_len = max(max_article_len, len(sample["article_token_ids"]))
            max_question_len = max(max_question_len, len(sample["question_token_ids"]))
            max_options_len = max(max_options_len, max([len(i) for i in sample["options_token_ids"]]))

            articles.append(sample['article_token_ids'])
            questions.append(sample['question_token_ids'])
            article_masks.append(sample['article_attention_mask'])
            question_masks.append(sample['question_attention_mask'])
            answer_indices.append(sample['answer_index'])
            options.append(sample['options_token_ids'])
            if self.config.split!="test":
                labels.append(label)

        for i in range(len(articles)):
            articles[i]= articles[i] + [self.pad_id]*(max_article_len - len(articles[i]))
            questions[i]= questions[i] + [self.pad_id]*(max_question_len - len(questions[i]))
            article_masks[i]= article_masks[i] + [self.pad_id]*(max_article_len - len(article_masks[i]))
            question_masks[i]= question_masks[i] + [self.pad_id]*(max_question_len - len(question_masks[i]))
            for option_index in range(len(options[i])):
                options[i][option_index] = options[i][option_index] + [self.pad_id]*(max_options_len -len(options[i][option_index]))


        return_dic = {
            'articles_token_ids':torch.LongTensor(articles),
            'questions_token_ids':torch.LongTensor(questions),
            'options_token_ids':torch.LongTensor(options),
            'answer_indices':torch.LongTensor(answer_indices),
            'articles_attention_mask':torch.FloatTensor(article_masks),
            'questions_attention_mask':torch.FloatTensor(question_masks),
        }

        if self.config.split=="test":
            return return_dic

        #return_dic['labels']=torch.LongTensor(labels)
        return return_dic, torch.LongTensor(labels)
