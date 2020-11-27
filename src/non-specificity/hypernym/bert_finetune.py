import yaml
import torch

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


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
        print("GPU not seen by Torch, please check again")

    # load the model
    print("\nLoading the model...")
    model = BertForMaskedLM.from_pretrained(args["model_name"])
    print(model.config)
    print("Number of parameters: " + str(model.num_parameters()))

    # invoke the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args["model_name"])

    # load the dataset
    print("\nLoading the dataset")
    t0 = time.time()

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=args["input_data_file"], block_size=512,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args["mlm_probability"]
    )

    print("Time taken: " + str(time.time() - t0))

    # training
    training_args = TrainingArguments(
        output_dir=args["save_model_directory"],
        overwrite_output_dir=True,
        num_train_epochs=args["num_train_epochs"],
        per_device_train_batch_size=args["batch_size"],
        save_steps=args["save_steps"],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    trainer.save_model(args["save_model_directory"])


if __name__ == "__main__":
    main()
