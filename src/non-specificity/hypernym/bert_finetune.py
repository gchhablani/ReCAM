import sys
import time


import yaml
import torch

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def freeze_bert_fn(
    model, freeze_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], freeze_embeddings=True
):

    if freeze_embeddings:
        for param in list(model.bert.embeddings.word_embeddings.parameters()):
            param.requires_grad = False
        print("Froze Embedding Layer")

    if len(freeze_layers) != 0:
        layer_indices = freeze_layers
        for layer_idx in layer_indices:
            for param in list(model.bert.encoder.layer[layer_idx].parameters()):
                # bert has 12 layers
                param.requires_grad = False
            print("Froze Layer: ", layer_idx)
            # print(model.bert.encoder.layer[layer_idx])

    return model


def main():

    # load the args from yaml file
    with open("bert_finetune.yaml") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    print("Printing Arguments...")
    for key in args:
        print("- " + str(key) + ": " + str(args[key]))

    print("\nDevice used...")
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("GPU not seen by Torch, please check again. Exiting for now...")
        exit()

    # load the model
    print("\nLoading the model...")
    model = BertForMaskedLM.from_pretrained(args["model_name"])
    # freeze the first 21 layers
    model = freeze_bert_fn(model, [i for i in range(0, 21)])  # freeze all but three
    print(model.config)
    print("Number of parameters: " + str(model.num_parameters()))

    # invoke the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args["model_name"])

    # load the dataset
    print("\nLoading the dataset")
    t0 = time.time()

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args["train_data_file"],
        block_size=512,
    )
    val_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args["val_data_file"],
        block_size=512,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args["mlm_probability"]
    )

    print("Time taken: " + str(time.time() - t0))

    # training
    training_args = TrainingArguments(
        output_dir=args["save_model_directory"],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=args["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=args["eval_steps"],
        num_train_epochs=args["num_train_epochs"],
        per_gpu_train_batch_size=args["batch_size"],
        per_gpu_eval_batch_size=args["batch_size"],
        save_steps=args["save_steps"],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    trainer.save_model(args["save_model_directory"])


if __name__ == "__main__":
    main()
