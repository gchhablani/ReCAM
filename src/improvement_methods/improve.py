import json
import argparse
from src.improvement_methods import StatisticalEmbedding
from src.datasets.cloze_dataset import ClozeDataset
from transformers import AutoTokenizer
from src.models import *
from src.utils.configuration import Config
import copy
from torch.utils.data import DataLoader
import torch
import heapq
import numpy as np

parser = argparse.ArgumentParser(
    prog="improve.py",
    description="Apply improvement approach to imperceptibility methods",
)

parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
    default=os.path.join(dirname, "./configs/models/forty/default.yaml"),
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
    default=os.path.join(dirname, "./configs/datasets/forty/default.yaml"),
)
parser.add_argument(
    "--trained_model_path",
    type=str,
    help="Path of the trained model's path",
    default="/content/drive/MyDrive/SemEval/SemEval_final/distilbert_train_trial/ReCAM-final/ckpts_old/all_ckpts/3_5600.pth",
)
parser.add_argument(
    "--test_configuration",
    help="Whether test data is being used.",
    type=str,
    default=False,
)
parser.add_argument(
    "--improvement_method",
    help="Select between: Thresholding Method(threshold), Difference Method(difference), Second Highest Probability Method(second_highest)",
    type=str,
    default="threshold",
)
dataset_path = Config(path=args.data)
model_config = Config(path=args.model)
path = args.trained_model_path
test_flag = args.test_configuration
emb = StatisticalEmbedding(normalise=False)


def generate_cloze_predictions(dataset_path, generate_hyponyms=False):
    with open(dataset_path) as f:
        datapoints = [json.loads(datapoint) for datapoint in f.read().splitlines()]
    model_name = model_config.params["pretrained_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cloze_dataset = ClozeDataset(dataset_config, tokenizer)
    weight = torch.load(path)
    model = torch.load(path)["model_state_dict"]
    dataloader = DataLoader(
        cloze_dataset,
        collate_fn=cloze_dataset.custom_collate_fn,
        batch_size=1,
        shuffle=False,
    )
    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            *inputs, label = [torch.tensor(value, device="cuda") for value in batch]
            datapoint = datapoints[i]
            if hyponym_run:
                cloze_prediction, bert_output = model(inputs)
                cloze_prediction_label = torch.argmax(cloze_prediction)
                final_datapoints.append(
                    {
                        "cloze_prediction": [
                            float(i) for i in cloze_prediction.cpu().numpy()[0]
                        ],
                        "cloze_prediction_label": int(
                            cloze_prediction_label.cpu().numpy()
                        ),
                        "datapoint": datapoint,
                        "bert_output": bert_output.detach().cpu().tolist(),
                    }
                )
            else:
                cloze_prediction = model(inputs)
                cloze_prediction_label = torch.argmax(cloze_prediction)
                final_datapoints.append(
                    {
                        "cloze_prediction": [
                            float(i) for i in cloze_prediction.cpu().numpy()[0]
                        ],
                        "cloze_prediction_label": int(
                            cloze_prediction_label.cpu().numpy()
                        ),
                        "datapoint": datapoint,
                    }
                )
            if test_flag:
                final_datapoints[-1]["id"] = datapoint["id"]
    return final_datapoints


def improvement_methods(p_value, final_datapoints, method="threshold"):
    lst_res = []
    softmax_function = torch.nn.Softmax()

    for i, data in enumerate(final_datapoints):
        cloze_preds = torch.Tensor(data["cloze_prediction"])
        cloze_probs = softmax_function(cloze_preds)
        if method == "threshold":
            pred_label = int(torch.argmax(cloze_probs))
            if cloze_probs[pred_label] < p_value:
                lst_inds = heapq.nlargest(
                    3, range(len(cloze_preds)), key=cloze_preds.__getitem__
                )
                second_max = lst_inds[1]
                second_max_option = data["datapoint"]["option_" + str(second_max)]
                max_option = data["datapoint"]["option_" + str(lst_inds[0])]
                max_opt_emb = np.array(emb.get_embedding(max_option))
                sec_opt_emb = np.array(emb.get_embedding(second_max_option))
                no_of_unequals = (
                    sec_opt_emb.shape[0] - (sec_opt_emb == max_opt_emb).sum()
                )
                if (sec_opt_emb > max_opt_emb).sum() >= no_of_unequals / 2:
                    final_datapoints[i]["cloze_prediction_label"] = second_max_option

        elif method == "difference":
            lst_inds = heapq.nlargest(
                3, range(len(cloze_preds)), key=cloze_preds.__getitem__
            )
            first_max = lst_inds[0]
            second_max = lst_inds[1]
            second_max_option = data["datapoint"]["option_" + str(second_max)]
            max_option = data["datapoint"]["option_" + str(first_max)]
            if cloze_probs[first_max] - cloze_probs[second_max] < p_value:
                max_opt_emb = np.array(emb.get_embedding(max_option))
                sec_opt_emb = np.array(emb.get_embedding(second_max_option))
                no_of_unequals = (
                    sec_opt_emb.shape[0] - (sec_opt_emb == max_opt_emb).sum()
                )
                if (sec_opt_emb > max_opt_emb).sum() >= no_of_unequals / 2:
                    final_datapoints[i]["cloze_prediction_label"] = second_max_option

        elif method == "second_highest":
            lst_inds = heapq.nlargest(
                3, range(len(cloze_preds)), key=cloze_preds.__getitem__
            )
            first_max = lst_inds[0]
            second_max = lst_inds[1]
            second_max_option = data["datapoint"]["option_" + str(second_max)]
            max_option = data["datapoint"]["option_" + str(first_max)]
            if cloze_probs[second_max] > p_value:
                max_opt_emb = np.array(emb.get_embedding(max_option))
                sec_opt_emb = np.array(emb.get_embedding(second_max_option))
                no_of_unequals = (
                    sec_opt_emb.shape[0] - (sec_opt_emb == max_opt_emb).sum()
                )
                if (sec_opt_emb > max_opt_emb).sum() >= no_of_unequals / 2:
                    final_datapoints[i]["cloze_prediction_label"] = second_max_option
    return final_datapoints


def write_to_csv(final_datapoints):
    output = ""
    for i, data in enumerate(final_datapoints):
        if test_flag:
            id = data["id"]
        else:
            id = i
        output += id + "," + int(data["cloze_prediction_label"]) + "\n"
    with open("output.csv", "w") as f:
        f.write(output)
