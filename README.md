# ReCAM
Our repository for the code, literature review, and results on SemEval-2021 Task 4:  Reading Comprehension of Abstract Meaning

## Tasks


### Imperceptibility

- Approaches:
	- Plain GAReader (Baseline)
	- Common-sense Reasoning
	- Transformers-Based Model for Cloze-Style Question Answering
	- Incorporating The Imperceptibility Rating from the Concreteness Rating Dataset
		- The dataset is present in ```data/Imperceptibility/Concreteness Ratings```.

### Non-Specificity

- Approaches: 
	- Use WordNet's hypernyms, augment the dataset (replace nouns with their hypernyms), pretrain BERT on this augmented dataset.
		- Code:

			```
			python -m nltk.downloader 'popular'
			pip install -U pywsd
			```
			and use the ```augment(sent)``` function in ```src/non-specificity/hypernym/augment.py```.

### References

#### Imperceptibility

- [Parameters of Abstraction, Meaningfulness, and Pronunciability for 329 Nouns](https://www.sciencedirect.com/science/article/abs/pii/S0022537166800610?via%3Dihub)
- [Gated-Attention Readers for Text Comprehension](https://arxiv.org/abs/1606.01549)
- [Concreteness Ratings for 40 Thousand Generally Known English Word Lemmas](https://www.researchgate.net/publication/258061778_Concreteness_ratings_for_40_thousand_generally_known_English_word_lemmas)

#### Non-Specificity
- [WorNet Interface (Hypernyms)](https://www.nltk.org/howto/wordnet.html)