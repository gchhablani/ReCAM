import glob
import nltk
from tqdm.auto import tqdm
from numpy.linalg import norm
import spacy
from nltk.corpus import sentiwordnet as swn
from itertools import chain
from nltk.corpus import wordnet as wn

nltk.download("punkt")
nlp = spacy.load("en", disable=["parser", "ner"])
nltk.download("sentiwordnet")
nltk.download("wordnet")


class StatisticalEmbedding:
    def __init__(self, normalise=True):
        # add word frequency later
        # try to fix number of senses and add it later
        # try to fix number of hyponyms and add it later
        self.normalise = normalise

    def get_embedding(self, word):
        len_embedding = self.get_length_of_word(word)
        sense_embedding = self.get_number_of_senses(word)
        hyponym_embedding = self.get_no_of_hyponyms(word)
        avg_hyponym_embedding = self.get_avg_no_of_hyponyms(word)
        depth_hypernymy_embedding = self.get_depth_of_hypernymy_tree(word)
        avg_depth_hypernymy_embedding = self.get_avg_depth_of_hypernymy_tree(word)
        pos_neg_obj_score = self.get_pos_neg_obj_scores(word)
        avg_pos_neg_obj_score = self.get_avg_pos_neg_obj_scores(word)

        embedding = [
            len_embedding,
            sense_embedding,
            hyponym_embedding,
            avg_hyponym_embedding,
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

    def get_number_of_senses(self, word):
        # words = word.split(' ')
        # lst_of_senses = [len(wn.synsets(word)) for word in words]
        # max_no_of_senses = max(lst_of_senses)
        return len(wn.synsets(word))

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

        return 100000 - max_len_paths

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
                return 100000 - avg_len_paths / i
            else:
                return 100000

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
        return (max(pos_scores), max(neg_scores), 1 - max(obj_scores))

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
        return (max(pos_scores), max(neg_scores), 1 - max(obj_scores))

    def get_no_of_hyponyms(self, word):

        if len(wn.synsets(word)) > 0:
            j = wn.synsets(word)[0]
            # print(word)
            # print(j.hyponyms())
            no_of_hypos = len(list(chain(*[l.lemma_names() for l in j.hyponyms()])))
            return no_of_hypos
        else:
            return 0

    def get_avg_no_of_hyponyms(self, word):
        i = 0
        no_of_hypos = 0
        for j in wn.synsets(word):
            no_of_hypos += len(list(chain(*[l.lemma_names() for l in j.hyponyms()])))
            i += 1
        if i > 0:
            return no_of_hypos / i
        else:
            return 0
