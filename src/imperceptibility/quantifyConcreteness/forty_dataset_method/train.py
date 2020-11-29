import argparse
import sys
import math
import os

sys.path.append("./src/imperceptibility/quantifyConcreteness/forty_dataset_method")


import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from transformers import AutoTokenizer, AdamW

from tqdm.auto import tqdm

from dataloader import ConcretenessDataset
from model import TwoLayerNN

from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

import pickle

writer = SummaryWriter("./logs_conc")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file_path",
        type=str,
        default="data/Imperceptibility/Concreteness Ratings/train/forty.csv",
        help="path of csv file",
    )

    parser.add_argument(
        "--val_file_path",
        type=str,
        default="data/Imperceptibility/Concreteness Ratings/val/forty.csv",
        help="path of csv file",
    )

    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="variant of BERT"
    )

    """
    parser.add_argument(
        "--w2v_path",
        type=str,
        default="data/Imperceptibility/W2V/GoogleNews-vectors-negative300.bin",
        help="path of Word2Vec file",
    )
    """

    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs for training"
    )

    parser.add_argument(
        "--val_after_every",
        type=int,
        default=20,
        help="validation loop after every x iterations",
    )

    parser.add_argument(
        "--tensorboard_label_number", type=str, default="1", help="logging number",
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")

    parser.add_argument(
        "--val_log_steps",
        type=int,
        default=100,
        help="validation after every x iterations",
    )

    args = parser.parse_args()

    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    model_name = args.model_name
    # w2v_path = args.w2v_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    learning_rate = args.lr
    val_log_steps = args.val_log_steps
    tensorboard_label_number = args.tensorboard_label_number

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    print("Loading data...")
    # train_set = ConcretenessDataset(csv_file=train_file_path, word_embedding=w2v_path)
    # val_set = ConcretenessDataset(csv_file=val_file_path, word_embedding=w2v_path)
    # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ConcretenessDataset(
        csv_file=train_file_path,
        batch_size=batch_size,
        tokenizer=tokenizer,
        split="training",
        device=device,
    )
    validation_dataset = ConcretenessDataset(
        csv_file=val_file_path,
        batch_size=batch_size,
        tokenizer=tokenizer,
        split="test",
        device=device,
    )

    print("Loading the model...")
    model = TwoLayerNN(model_name)
    model.to(device)
    # model.cuda()

    # print model summary
    # sample = tokenizer("I am bored",return_tensors="pt")
    # sample['labels'] = torch.tensor([1])
    # print(summary(model, sample.to(device)))

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    """

    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        tr_loss = 0
        val_loss = 0
        pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
        pbar.set_description("Epoch " + str(epoch))
        i = 1
        val_counter = 0
        # Training Loop
        while True:
            optimizer.zero_grad()

            batch = train_dataset.load_batch()
            if batch is None:
                break
            # print(batch['input_ids'].shape)
            inputs, labels = batch
            # labels = inputs['labels']
            output = model(inputs)
            # print(output[1].shape)

            loss = loss_fn(output, labels)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()

            pbar.set_postfix_str("Loss: " + str(tr_loss / i))
            pbar.update(1)
            # print("[train] loss: {}".format(tr_loss / steps))
            # steps += 1
            i += 1
            if i % val_log_steps == 0:
                j = 1
                val_loss = 0
                print("\nChecking Validation Loss...")
                model.eval()
                # y_true = []
                # y_pred = []
                # out_score = []
                total = len(validation_dataset) // batch_size
                while True:
                    with torch.no_grad():
                        batch = validation_dataset.load_batch()
                        if batch is None:
                            break
                        inputs, labels = batch
                        # label_tensor = inputs['labels']

                        output = model(inputs)

                        # out_score += [p for p in output.flatten().tolist()]
                        loss = loss_fn(output, labels)
                        val_loss += loss.item()
                        pbar.set_postfix_str(
                            "Val Loss: "
                            + str(val_loss / j)
                            + "\tProgress:"
                            + str(j)
                            + "/"
                            + str(total)
                        )

                        j += 1
                val_counter += j

                writer.add_scalar(
                    "training loss"
                    + str(tensorboard_label_number)
                    + str(learning_rate),
                    tr_loss / i,
                    epoch * len(train_dataset) + i,
                )

                writer.add_scalar(
                    "validation loss"
                    + str(tensorboard_label_number)
                    + str(learning_rate),
                    val_loss / j,
                    epoch * len(validation_dataset) + j + val_counter,
                )

                print("\n Switching back to training...")
                model.train()

        pbar.close()
        # Validation Loop

        if not os.path.exists("model_ckpts"):
            os.makedirs("model_ckpts")

        torch.save(
            model.state_dict(),
            "model_ckpts/ckpt_"
            + str(tensorboard_label_number)
            + "_"
            + str(epoch)
            + ".pth",
        )


if __name__ == "__main__":
    main()
