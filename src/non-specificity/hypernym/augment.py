import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk
import random
import itertools


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def hypernyms(sent):

    # find all nouns in text (POS Tagging)
    text = nltk.word_tokenize(sent)
    pos_text = nltk.pos_tag(text)

    # find hypernyms
    hypernyms = {}
    lst_of_words = []
    for w in pos_text:
        if w[1] == "NN":
            # apply LESK algorithm to get the correct word sense (WSD)
            answer = simple_lesk(sent, w[0], pos="n")

            if answer:
                hyp = answer.hypernyms()
                lst_of_words.append(w[0])
                hypernyms[w[0]] = hyp

    return hypernyms, lst_of_words


def augment(sent):
    hypernyms_dict, lst_of_words = hypernyms(sent)
    augmented_sentences = []

    for n in range(1, len(lst_of_words) + 1):
        words = findsubsets(lst_of_words, n)
        new_sent = "".join(sent)
        for word in words:
            i = random.randint(0,len(hypernyms_dict[word])-1)
            new_phrase = (
                str(hypernyms_dict[word][i])
                .split("(")[1]
                .replace(")", "")
                .replace("'", "")
                .split(".")[0]
                .replace("_", " ")
            )
            new_sent = new_sent.replace(word, new_phrase)
        augmented_sentences.append(new_sent)

    return augmented_sentences
