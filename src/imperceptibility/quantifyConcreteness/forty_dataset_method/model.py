# Implements a two layer Neural Network

import torch
from transformers import BertModel


class TwoLayerNN(torch.nn.Module):
    def __init__(self, model_name):

        super(TwoLayerNN, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, H2)
        # self.linear3 = torch.nn.Linear(H2, D_out)
        # self.relu = torch.nn.ReLU()
        # self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=1)
        """
        for name, param in self.model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
        """
        # defined this way because it will be easier later to append the statistical features
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)

        # freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear1 = torch.nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)

        """
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        """

    def forward(self, x):

        # x1 = self.relu(self.linear1(x))
        # x2 = self.relu(self.linear2(x1))
        # print(x)
        # x_output = self.linear3(x2)
        # x_output = self.model(**x)

        bert_output = self.bert(**x)
        last_hidden_state_cls = bert_output.last_hidden_state[:, 0, :]
        output = self.linear1(last_hidden_state_cls)
        output = self.relu(output)
        x_output = self.linear2(output)
        # x_output = torch.sigmoid(x_output)
        return x_output
