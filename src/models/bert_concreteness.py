import torch
from transformers import BertModel, BertForSequenceClassification


@configmapper.map("models", "bert_concreteness")
class BERTConcreteness(torch.nn.Module):

    """
        Implements BERT with a Linear Layer on top to perform the task of regression on the Concreteness Dataset.

        Methods:
            forward(x): Returns the output of the Neural Network
    """

    def __init__(self, model_name="bert-base-uncased", freeze_layers=10):

        """
            Constructs the Neural Network---BERT with Linear Layer atop [CLS]'s embedding

            Args:
                model_name (str): One of "bert-base-uncased"/"bert-large-uncased"
                freeze_layers (int): Number of BERT encoder layers to freeze
        """

        super(BERTConcreteness, self).__init__()

        assert (model_name == "bert-base-uncased" and freeze_layers <= 12) or (
            model_name == "bert-large-uncased" and freeze_layers <= 24
        ), "model_name should be either bert-base-uncased with freeze_layers<=12/bert-large-uncased with freeze_layers<=24"

        self.bert = BertForSequenceClassification.from_pretrained(
            model_name, return_dict=True, num_labels=1
        )

        # freeze freeze_layers number of layers
        freeze_layers = [i for i in range(freeze_layers)]
        self.bert = self.freeze_bert_fn(self.bert, freeze_layers, True)

    def forward(self, x):

        """
            Return the output of the neural network for an input.
            
            Args:
                x (dict): The input dict to the neural network (see input format of BertForSequenceClassification on https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
            
            Returns:
                output (transformers.modeling_outputs.SequenceClassifierOutput): loss/hidden_states, etc. 
        """

        output = self.bert(**x)
        return output

    def freeze_bert_fn(
        self,
        model,
        freeze_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        freeze_embeddings=True,
    ):

        """
            Returns the model with the top "freeze_layers" layers frozen.

            Args:
                model (transformers.models.bert.modeling_bert.BertForSequenceClassification): model whose layers are to be frozen
                freeze_layers (list): The layers to be frozen
                freeze_embeddings (boolean)

            Returns:
                model (transformers.models.bert.modeling_bert.BertForSequenceClassification) with frozen layers
        """

        if freeze_embeddings:
            for param in list(model.bert.embeddings.word_embeddings.parameters()):
                param.requires_grad = False
            print("Froze Embedding Layer")

        if len(freeze_layers) != 0:
            layer_indices = freeze_layers
            for layer_idx in layer_indices:
                for param in list(model.bert.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                print("Froze Layer: ", layer_idx)

        return model
