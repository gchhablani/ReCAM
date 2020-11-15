'''
    TASK for Script => 
    converts from : dict_keys(['article', 'question', 'option_0', 'option_1', 'option_2', 'option_3', 'option_4', 'label'])
    to : dict_keys(['article', 'options', 'answers', 'source'])
    where article_original = "some text. question @placeholder question?"
    to article_modified = "some text. question _ question?"
    where eg. "options": [
        [
            "between", 
            "before", 
            "since", 
            "later"
        ], 
        [
            "after", 
            "by", 
            "during", 
            "until"
        ], ... ] 
    Divide each question into 1 json file , into seperated folders
'''
import argparse
import json
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/training_data/Task_1_train.jsonl",
        help="Path of the jsonl containing data samples.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Where the output json files should be stored. Format: .../def",
    )
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    #reading jsonl file
    with open(data_path) as f:
        datapoints = [json.loads(datapoint) for datapoint in f.read().splitlines()]

    #converting to correct format
    for index,datapoint in enumerate(datapoints):
        question = datapoint['question'].replace('@placeholder','_')
        article = datapoint['article'].replace('_','') + " " + question
        options = [[datapoint['option_0'],datapoint['option_1'],datapoint['option_2'],datapoint['option_3'],datapoint['option_4']]]
        answers = [chr(datapoint['label']+65)]
        source = "high"+str(index)
        final_output = {'article':article,'options':options,'answers':answers,'source':source}
        with open(output_path+'/'+source+'.json','w') as f:
            json.dump(final_output,f)


if __name__ == "__main__":
    main()