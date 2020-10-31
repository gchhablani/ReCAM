import argparse
import sys

sys.path.append("./src/imperceptibility/quantifyConcreteness")


import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn

from tqdm.auto import tqdm

from dataloader import ConcretenessDataset
from model import TwoLayerNN


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file_path",
        type=str,
        default="data/Imperceptibility/Concreteness Ratings/train/AC_ratings_google3m_koeper_SiW.csv",
        help="path of csv file",
    )

    parser.add_argument(
        "--val_file_path",
        type=str,
        default="data/Imperceptibility/Concreteness Ratings/val/AC_ratings_google3m_koeper_SiW.csv",
        help="path of csv file",
    )

    parser.add_argument(
        "--w2v_path",
        type=str,
        default="data/Imperceptibility/W2V/GoogleNews-vectors-negative300.bin",
        help="path of Word2Vec file",
    )

    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training"
    )

    parser.add_argument(
        "--val_after_every",
        type=int,
        default=20,
        help="validation loop after every x iterations",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    args = parser.parse_args()

    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    w2v_path = args.w2v_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    print("Loading data...")
    train_set = ConcretenessDataset(csv_file=train_file_path, word_embedding=w2v_path)
    val_set = ConcretenessDataset(csv_file=val_file_path, word_embedding=w2v_path)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print("Loading the model...")
    model = TwoLayerNN()
    model.to(device)
    # model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_tqdm = tqdm(dataloader)
    val_loader_tqdm = tqdm(val_loader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        i = 1
        for data in dataloader_tqdm:

            dataloader_tqdm.set_description_str("Epoch: " + str(epoch + 1))

            inputs, score = data
            inputs = torch.tensor(inputs).float().to(device)
            score = torch.tensor(score).float().to(device)
            # inputs = torch.tensor(inputs).float().cuda()
            # score = torch.tensor(score).float().cuda()
            # print(inputs.shape)
            # print(score.shape)
            # print(inputs)
            # print(score)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, score)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            dataloader_tqdm.set_postfix_str("LOSS: " + str(running_loss / i))
            i += 1

            if i % args.val_after_every == 0:
                val_running_loss = 0.0
                j = 0
                with torch.no_grad():
                    for val_data in val_loader_tqdm:
                        inputs_val, score_val = val_data
                        inputs_val = torch.tensor(inputs_val).float().to(device)
                        score_val = torch.tensor(score_val).float().to(device)
                        # inputs = torch.tensor(inputs).float().cuda()
                        # score = torch.tensor(score).float().cuda()
                        # print(inputs.shape)
                        # print(score.shape)

                        outputs_val = model(inputs_val)
                        loss_val = criterion(outputs_val, score_val)
                        val_running_loss += loss_val.item()
                        dataloader_tqdm.set_postfix_str(
                            "VAL LOSS: " + str(val_running_loss / i)
                        )


if __name__ == "__main__":
    main()
