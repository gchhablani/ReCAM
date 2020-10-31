import argparse
import sys
sys.path.append('./src/imperceptibility/quantifyConcreteness')


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
        "--csv_file_path",
        type=str,
        default="data/Imperceptibility/Concreteness Ratings/AC_ratings_google3m_koeper_SiW.csv",
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

    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")

    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    w2v_path = args.w2v_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device))



    print("Loading data...")
    train_set = ConcretenessDataset(csv_file=csv_file_path, word_embedding=w2v_path)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print("Loading the model...")
    model = TwoLayerNN()
    model.to(device)
    #model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_tqdm = tqdm(dataloader)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        i = 1
        for data in dataloader_tqdm:


            dataloader_tqdm.set_description_str("Epoch: " + str(epoch + 1))
            
            inputs, score = data
            inputs = torch.tensor(inputs).float().to(device)
            score = torch.tensor(score).float().to(device)
            #inputs = torch.tensor(inputs).float().cuda()
            #score = torch.tensor(score).float().cuda()
            #print(inputs.shape)
            #print(score.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, score)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            dataloader_tqdm.set_postfix_str("LOSS: " + str(running_loss / i))
            i += 1


if __name__ == "__main__":
    main()
