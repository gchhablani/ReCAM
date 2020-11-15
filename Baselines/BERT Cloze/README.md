#Steps to run project:
1) Create a folder CLOZE, with train and valid as sub-folders
2) Create a sub-folder "high" in both, train and valid folders
3) Run : !python bert_cloze_data_script.py --data_path path_to_train_jsonl_data --output_path ./CLOZE/train/high
4) Run : !python bert_cloze_data_script.py --data_path path_to_valid_jsonl_data --output_path ./CLOZE/valid/high
5) Run : python data_util.py
6) Run : ./run.sh (in run.sh we can change details like batch size, bert model type, learning rate, etc.)
