import spacy
import time
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

import torch

from torchtext import data
from torchtext import datasets
from torchtext import vocab

NLP = spacy.blank("en")


def word_tokenize(sent):
    """ Tokenize """
    doc = NLP(sent)
    return [token.text for token in doc]


def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def classifiction_metric(preds, labels, label_list):
    """ The Metric of classification, input should be numpy format """

    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(
        labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)

    return acc, report

