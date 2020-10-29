# ReCAM
Our repository for the code, literature review, and results on SemEval-2021 Task 4:  Reading Comprehension of Abstract Meaning

## Tasks


### Imperceptibility

- Approach 1:
	- Plain GAReader (Baseline)
	- Common-sense Reasoning
	- Transformers-Based Model for Cloze-Style Question Answering
	- Incorporating The Imperceptibility Rating from the Concreteness Rating Dataset
		- The dataset is present ```data/Imperceptibility/Concreteness Ratings```.

### Non-Specificity

- Approach 1: 
	- Use WordNet's hypernyms, augment the dataset (replace nouns with their hypernyms), pretrain BERT on this augmented dataset.
		- Code:

			```
			python -m nltk.downloader 'popular'
			pip install -U pywsd
			```
			and use the ```augment(sent)``` function in ```src/non-specificity/hypernym/augment.py```.

