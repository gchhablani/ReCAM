"""Evaluate some model."""

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.concreteness_dataset import ConcretenessDataset
from models.two_layer_nn import TwoLayerNN
from utils.embeddings import GloveEmbedding
from utils.tokenizers import GloveTokenizer

train_file_path = "../data/imperceptibility/Concreteness Ratings/train/forty.csv"
val_file_path = "../data/imperceptibility/Concreteness Ratings/val/forty.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GloveTokenizer(cache="../embeddings/")
tokenizer.initialize_vectors(
    tokenizer_file_paths=[train_file_path, val_file_path], fields=["Word"]
)


validation_dataset = ConcretenessDataset(
    file_path=val_file_path,
    tokenizer=tokenizer,
    split="val",
)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
embeddings = GloveEmbedding(
    tokenizer.text_field.vocab.vectors,
    tokenizer.text_field.vocab.stoi[tokenizer.text_field.pad_token],
)
dims = [1200, 128, 1]
model = TwoLayerNN(embeddings, dims)
model.load_state_dict(torch.load("../model_ckpts/ckpt_1_10.pth"))


model.eval()
# y_true = []
# y_pred = []
# out_score = []
total = len(validation_dataset) // 1
for batch in val_loader:
    with torch.no_grad():
        inputs, labels = batch
        inputs.to(device)
        labels.to(device)
        labels = labels.float()
        # label_tensor = inputs['labels']

        output = model(inputs)
        print(torch.squeeze(output), labels)
