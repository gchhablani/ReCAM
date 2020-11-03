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

## To-Do                
- [x] Exploratory Data Analysis
  - [x] Find Most Common Words
  - [ ] The GloVe/CoVe representation of words in passage, words in question, and words in answer (Couple of examples).
  - [x] Word Split Based EDA (Stats)
  - [x] Bert Tokenizer Based EDA (Stats)
  - [ ] WordClouds

- [ ] Build Basic Q/A system
	- [ ] GA Reader

- [ ] Literature Reviews
## Data
### Statistics

#### Word/Char Python Split-Based
|                             | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:----------------------------|:-------------|:-----------|:-------------|:-----------|
| Samples                     | 3227         | 837        | 3318         | 851        |
| Duplicated Examples         | 0            | 0          | 0            | 0          |
| Duplicated Articles         | 57           | 4          | 55           | 2          |
| Mean Char Count (Articles)  | 1548.47      | 1582.79    | 2448.93      | 2501.41    |
| Std Char Count (Articles)   | 905.301      | 992.333    | 1788.42      | 1809.55    |
| Max Char Count (Articles)   | 10177        | 9563       | 10370        | 9628       |
| Min Char Count (Articles)   | 162          | 255        | 169          | 247        |
| Mean Word Count (Articles)  | 262.44       | 268.608    | 417.727      | 427.048    |
| Std Word Count (Articles)   | 154.534      | 171.253    | 307.248      | 310.994    |
| Max Word Count (Articles)   | 1754         | 1641       | 1700         | 1651       |
| Min Word Count (Articles)   | 31           | 41         | 31           | 45         |
| Duplicated Questions        | 0            | 0          | 1            | 0          |
| Mean Char Count (Questions) | 137.826      | 137.455    | 148.234      | 149.385    |
| Std Char Count (Questions)  | 31.1171      | 30.6809    | 47.7494      | 46.9035    |
| Max Char Count (Questions)  | 389          | 387        | 443          | 413        |
| Min Char Count (Questions)  | 32           | 48         | 31           | 37         |
| Mean Word Count (Questions) | 24.6771      | 24.4695    | 26.8852      | 27.2127    |
| Std Word Count (Questions)  | 6.18453      | 6.04612    | 9.1822       | 9.16159    |
| Max Word Count (Questions)  | 73           | 73         | 83           | 83         |
| Min Word Count (Questions)  | 6            | 9          | 7            | 6          |

#### BertTokenizer Based
|                              | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:-----------------------------|:-------------|:-----------|:-------------|:-----------|
| Mean Token Count (Articles)  | 332.603      | 341.452    | 528.58       | 539.303    |
| Std Token Count (Articles)   | 193.729      | 217.673    | 387.756      | 396.286    |
| Max Token Count (Articles)   | 2206         | 2043       | 2188         | 2272       |
| Min Token Count (Articles)   | 38           | 52         | 46           | 52         |
| Mean Token Count (Questions) | 28.6319      | 28.3441    | 30.9629      | 31.1657    |
| Std Token Count (Questions)  | 6.83765      | 6.75112    | 10.0225      | 9.98314    |
| Max Token Count (Questions)  | 84           | 84         | 91           | 89         |
| Min Token Count (Questions)  | 9            | 11         | 9            | 8          |

**Note** :
- Word is based on .split() on Python strings, including all the words (stopwords).
We can also use torch/nltk tokenizers to get tokens and check their lengths.
- The spaces are also included in character counting.
- No preprocessing was done for either of the statistics.
- For questions, we do not remove @placeholder.


### Most Frequent Words
**Note** :
- Tokens were made using NLTK Tokenizer, and only alphanumeric characters, after removing stop words were used.

#### Articles
|                    | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:-------------------|:-------------|:-----------|:-------------|:-----------|
| Unique Token Count | 41087        | 20447      | 53285        | 25779      |
| Total Count        | 482713       | 128258     | 775342       | 203877     |

|    | Task_1_train |           |               | Task_1_dev |           |               | Task_2_train |           |               | Task_2_dev |           |               |
|:---|:-------------|:----------|:--------------|:-----------|:----------|:--------------|:-------------|:----------|:--------------|:-----------|:----------|:--------------|
|    | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** |
| 0  | said         | 7978      | 0.0165274     | said       | 1940      | 0.0151258     | said         | 9594      | 0.0123739     | said       | 2524      | 0.01238       |
| 1  | mr           | 2465      | 0.00510655    | mr         | 701       | 0.00546555    | mr           | 3549      | 0.00457733    | mr         | 993       | 0.00487058    |
| 2  | would        | 2194      | 0.00454514    | would      | 593       | 0.00462349    | would        | 3539      | 0.00456444    | would      | 954       | 0.00467929    |
| 3  | also         | 2019      | 0.00418261    | also       | 506       | 0.00394517    | people       | 3335      | 0.00430133    | people     | 872       | 0.00427709    |
| 4  | people       | 1897      | 0.00392987    | one        | 486       | 0.00378924    | one          | 3182      | 0.004104      | one        | 814       | 0.0039926     |
| 5  | one          | 1705      | 0.00353212    | people     | 481       | 0.00375025    | also         | 2784      | 0.00359067    | also       | 675       | 0.00331082    |
| 6  | last         | 1461      | 0.00302664    | first      | 395       | 0.00307973    | says         | 2201      | 0.00283875    | two        | 562       | 0.00275656    |
| 7  | two          | 1349      | 0.00279462    | last       | 391       | 0.00304854    | two          | 2112      | 0.00272396    | says       | 547       | 0.00268299    |
| 8  | first        | 1344      | 0.00278426    | new        | 372       | 0.0029004     | years        | 2084      | 0.00268785    | new        | 545       | 0.00267318    |
| 9  | new          | 1341      | 0.00277805    | told       | 349       | 0.00272108    | new          | 2081      | 0.00268398    | time       | 532       | 0.00260942    |
| 10 | year         | 1262      | 0.00261439    | two        | 346       | 0.00269769    | time         | 2064      | 0.00266205    | us         | 531       | 0.00260451    |
| 11 | years        | 1248      | 0.00258539    | us         | 342       | 0.0026665     | could        | 2002      | 0.00258209    | could      | 517       | 0.00253584    |
| 12 | could        | 1222      | 0.00253152    | years      | 340       | 0.00265091    | last         | 1957      | 0.00252405    | years      | 516       | 0.00253094    |
| 13 | told         | 1203      | 0.00249216    | year       | 329       | 0.00256514    | first        | 1944      | 0.00250728    | first      | 509       | 0.0024966     |
| 14 | us           | 1187      | 0.00245902    | time       | 307       | 0.00239361    | us           | 1820      | 0.00234735    | told       | 493       | 0.00241812    |
| 15 | time         | 1159      | 0.00240101    | could      | 291       | 0.00226886    | year         | 1768      | 0.00228028    | year       | 462       | 0.00226607    |
| 16 | government   | 1036      | 0.0021462     | bbc        | 256       | 0.00199598    | like         | 1727      | 0.0022274     | last       | 455       | 0.00223174    |
| 17 | made         | 940       | 0.00194733    | added      | 246       | 0.00191801    | told         | 1659      | 0.0021397     | bbc        | 418       | 0.00205026    |
| 18 | bbc          | 894       | 0.00185203    | made       | 236       | 0.00184004    | police       | 1504      | 0.00193979    | like       | 408       | 0.00200121    |
| 19 | police       | 892       | 0.00184789    | get        | 227       | 0.00176987    | get          | 1476      | 0.00190368    | government | 404       | 0.00198159    |

#### Questions
|                    | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:-------------------|:-------------|:-----------|:-------------|:-----------|
| Unique Token Count | 10032        | 4602       | 11098        | 4966       |
| Total Count        | 43683        | 11289      | 47297        | 12255      |

|    | Task_1_train |           |               | Task_1_dev  |           |               | Task_2_train |           |               | Task_2_dev  |           |               |
|:---|:-------------|:----------|:--------------|:------------|:----------|:--------------|:-------------|:----------|:--------------|:------------|:----------|:--------------|
|    | **Word**     | **Count** | **Frequency** | **Word**    | **Count** | **Frequency** | **Word**     | **Count** | **Frequency** | **Word**    | **Count** | **Frequency** |
| 0  | placeholder  | 3227      | 0.0738731     | placeholder | 837       | 0.074143      | placeholder  | 3318      | 0.0701524     | placeholder | 851       | 0.069441      |
| 1  | said         | 268       | 0.00613511    | new         | 69        | 0.00611214    | new          | 229       | 0.00484174    | new         | 71        | 0.00579355    |
| 2  | year         | 230       | 0.00526521    | says        | 62        | 0.00549207    | year         | 218       | 0.00460917    | two         | 60        | 0.00489596    |
| 3  | new          | 222       | 0.00508207    | year        | 56        | 0.00496058    | two          | 209       | 0.00441888    | says        | 57        | 0.00465116    |
| 4  | says         | 197       | 0.00450976    | us          | 48        | 0.00425193    | one          | 201       | 0.00424974    | year        | 54        | 0.00440636    |
| 5  | two          | 172       | 0.00393746    | said        | 46        | 0.00407476    | said         | 200       | 0.0042286     | said        | 54        | 0.00440636    |
| 6  | first        | 159       | 0.00363986    | first       | 43        | 0.00380902    | man          | 191       | 0.00403831    | us          | 49        | 0.00399837    |
| 7  | world        | 158       | 0.00361697    | two         | 43        | 0.00380902    | police       | 185       | 0.00391145    | people      | 48        | 0.00391677    |
| 8  | one          | 148       | 0.00338805    | world       | 38        | 0.00336611    | people       | 177       | 0.00374231    | one         | 47        | 0.00383517    |
| 9  | man          | 143       | 0.00327358    | uk          | 35        | 0.00310036    | first        | 174       | 0.00367888    | man         | 46        | 0.00375357    |
| 10 | people       | 141       | 0.0032278     | one         | 34        | 0.00301178    | world        | 168       | 0.00355202    | first       | 44        | 0.00359037    |
| 11 | us           | 129       | 0.00295309    | man         | 32        | 0.00283462    | years        | 154       | 0.00325602    | years       | 37        | 0.00301918    |
| 12 | police       | 127       | 0.00290731    | people      | 32        | 0.00283462    | us           | 147       | 0.00310802    | police      | 36        | 0.00293758    |
| 13 | uk           | 127       | 0.00290731    | police      | 31        | 0.00274604    | uk           | 133       | 0.00281202    | uk          | 32        | 0.00261118    |
| 14 | wales        | 118       | 0.00270128    | former      | 31        | 0.00274604    | says         | 130       | 0.00274859    | world       | 31        | 0.00252958    |
| 15 | years        | 115       | 0.0026326     | england     | 31        | 0.00274604    | three        | 113       | 0.00238916    | could       | 31        | 0.00252958    |
| 16 | former       | 112       | 0.00256393    | wales       | 29        | 0.00256887    | england      | 111       | 0.00234687    | city        | 28        | 0.00228478    |
| 17 | government   | 110       | 0.00251814    | league      | 29        | 0.00256887    | old          | 105       | 0.00222001    | found       | 27        | 0.00220318    |
| 18 | could        | 109       | 0.00249525    | years       | 28        | 0.00248029    | could        | 102       | 0.00215658    | may         | 27        | 0.00220318    |
| 19 | city         | 106       | 0.00242657    | three       | 25        | 0.00221455    | last         | 100       | 0.0021143     | bbc         | 27        | 0.00220318    |

**Note**:
- BERT Tokenizer based splits
	- Process - Take passages, convert to lowercase, remove all non-alphanumeric characters, tokenize, then remove stop words.

#### Articles

|                    | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:-------------------|:-------------|:-----------|:-------------|:-----------|
| Unique Token Count | 21952        | 15858      | 23842        | 18157      |
| Total Count        | 559481       | 148611     | 898160       | 235177     |

|    | Task_1_train |           |               | Task_1_dev |           |               | Task_2_train |           |               | Task_2_dev |           |               |
|:---|:-------------|:----------|:--------------|:-----------|:----------|:--------------|:-------------|:----------|:--------------|:-----------|:----------|:--------------|
|    | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** |
| 0  | ##s          | 8094      | 0.014467      | ##s        | 2393      | 0.0161024     | ##s          | 13511     | 0.015043      | ##s        | 3477      | 0.0147846     |
| 1  | said         | 7994      | 0.0142882     | said       | 1941      | 0.0130609     | said         | 9597      | 0.0106852     | said       | 2525      | 0.0107366     |
| 2  | mr           | 2466      | 0.00440766    | mr         | 702       | 0.00472374    | ##t          | 4282      | 0.00476752    | ##t        | 1068      | 0.00454126    |
| 3  | ##t          | 2266      | 0.00405018    | ##t        | 674       | 0.00453533    | mr           | 3550      | 0.00395253    | mr         | 993       | 0.00422235    |
| 4  | would        | 2204      | 0.00393937    | would      | 595       | 0.00400374    | would        | 3549      | 0.00395141    | would      | 964       | 0.00409904    |
| 5  | also         | 2020      | 0.00361049    | one        | 513       | 0.00345197    | one          | 3348      | 0.00372762    | people     | 873       | 0.0037121     |
| 6  | people       | 1897      | 0.00339064    | also       | 506       | 0.00340486    | people       | 3335      | 0.00371315    | one        | 855       | 0.00363556    |
| 7  | one          | 1822      | 0.00325659    | people     | 481       | 0.00323664    | also         | 2786      | 0.0031019     | also       | 675       | 0.00287018    |
| 8  | two          | 1479      | 0.00264352    | first      | 408       | 0.00274542    | two          | 2277      | 0.00253518    | two        | 611       | 0.00259804    |
| 9  | last         | 1477      | 0.00263995    | last       | 399       | 0.00268486    | says         | 2201      | 0.00245057    | new        | 557       | 0.00236843    |
| 10 | first        | 1417      | 0.0025327     | new        | 380       | 0.00255701    | new          | 2129      | 0.0023704     | says       | 547       | 0.00232591    |
| 11 | new          | 1372      | 0.00245227    | two        | 379       | 0.00255028    | years        | 2084      | 0.0023203     | us         | 546       | 0.00232166    |
| 12 | year         | 1281      | 0.00228962    | told       | 349       | 0.00234841    | time         | 2076      | 0.00231139    | time       | 540       | 0.00229614    |
| 13 | years        | 1248      | 0.00223064    | us         | 345       | 0.0023215     | first        | 2056      | 0.00228912    | first      | 525       | 0.00223236    |
| 14 | could        | 1224      | 0.00218774    | years      | 340       | 0.00228785    | could        | 2007      | 0.00223457    | could      | 518       | 0.0022026     |
| 15 | ##m          | 1219      | 0.0021788     | ##e        | 339       | 0.00228112    | last         | 1975      | 0.00219894    | years      | 516       | 0.00219409    |
| 16 | us           | 1206      | 0.00215557    | year       | 330       | 0.00222056    | us           | 1844      | 0.00205309    | told       | 493       | 0.00209629    |
| 17 | told         | 1203      | 0.00215021    | bbc        | 328       | 0.0022071     | year         | 1793      | 0.0019963     | bbc        | 475       | 0.00201976    |
| 18 | time         | 1173      | 0.00209659    | time       | 311       | 0.00209271    | ##ing        | 1785      | 0.0019874     | ##e        | 472       | 0.002007      |
| 19 | bbc          | 1112      | 0.00198756    | could      | 291       | 0.00195813    | ##e          | 1761      | 0.00196068    | year       | 468       | 0.00198999    |

#### Questions
|                    | Task_1_train | Task_1_dev | Task_2_train | Task_2_dev |
|:-------------------|:-------------|:-----------|:-------------|:-----------|
| Unique Token Count | 9771         | 4876       | 10702        | 5251       |
| Total Count        | 51629        | 13284      | 55858        | 14331      |

|    | Task_1_train |           |               | Task_1_dev |           |               | Task_2_train |           |               | Task_2_dev |           |               |
|:---|:-------------|:----------|:--------------|:-----------|:----------|:--------------|:-------------|:----------|:--------------|:-----------|:----------|:--------------|
|    | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** | **Word**     | **Count** | **Frequency** | **Word**   | **Count** | **Frequency** |
| 0  | place        | 3256      | 0.0630653     | place      | 844       | 0.0635351     | place        | 3349      | 0.0599556     | place      | 859       | 0.05994       |
| 1  | ##holder     | 3227      | 0.0625036     | ##holder   | 837       | 0.0630081     | ##holder     | 3318      | 0.0594006     | ##holder   | 851       | 0.0593818     |
| 2  | said         | 272       | 0.00526836    | new        | 69        | 0.00519422    | ##s          | 333       | 0.00596155    | new        | 72        | 0.00502407    |
| 3  | year         | 230       | 0.00445486    | ##s        | 66        | 0.00496838    | new          | 230       | 0.00411758    | ##s        | 72        | 0.00502407    |
| 4  | ##s          | 223       | 0.00431928    | says       | 62        | 0.00466727    | year         | 218       | 0.00390275    | two        | 60        | 0.00418673    |
| 5  | new          | 222       | 0.00429991    | year       | 56        | 0.0042156     | two          | 209       | 0.00374163    | says       | 57        | 0.00397739    |
| 6  | says         | 197       | 0.00381568    | us         | 48        | 0.00361337    | one          | 201       | 0.00359841    | year       | 54        | 0.00376806    |
| 7  | two          | 172       | 0.00333146    | said       | 46        | 0.00346281    | said         | 200       | 0.00358051    | said       | 54        | 0.00376806    |
| 8  | first        | 160       | 0.00309903    | first      | 43        | 0.00323698    | man          | 196       | 0.0035089     | man        | 50        | 0.00348894    |
| 9  | world        | 158       | 0.0030603     | two        | 43        | 0.00323698    | police       | 185       | 0.00331197    | us         | 49        | 0.00341916    |
| 10 | man          | 150       | 0.00290534    | world      | 38        | 0.00286058    | people       | 177       | 0.00316875    | people     | 48        | 0.00334938    |
| 11 | one          | 148       | 0.00286661    | uk         | 36        | 0.00271003    | first        | 174       | 0.00311504    | one        | 47        | 0.0032796     |
| 12 | people       | 141       | 0.00273102    | man        | 35        | 0.00263475    | world        | 168       | 0.00300763    | first      | 44        | 0.00307027    |
| 13 | uk           | 135       | 0.00261481    | one        | 34        | 0.00255947    | years        | 154       | 0.00275699    | years      | 37        | 0.00258182    |
| 14 | us           | 130       | 0.00251796    | people     | 32        | 0.00240891    | us           | 147       | 0.00263167    | police     | 36        | 0.00251204    |
| 15 | police       | 129       | 0.0024986     | police     | 31        | 0.00233363    | uk           | 142       | 0.00254216    | uk         | 33        | 0.0023027     |
| 16 | wales        | 118       | 0.00228554    | former     | 31        | 0.00233363    | says         | 130       | 0.00232733    | world      | 32        | 0.00223292    |
| 17 | years        | 115       | 0.00222743    | england    | 31        | 0.00233363    | three        | 113       | 0.00202299    | could      | 31        | 0.00216314    |
| 18 | former       | 112       | 0.00216932    | wales      | 29        | 0.00218308    | england      | 111       | 0.00198718    | city       | 28        | 0.00195381    |
| 19 | government   | 110       | 0.00213059    | league     | 29        | 0.00218308    | old          | 105       | 0.00187977    | found      | 27        | 0.00188403    |
### References

#### Imperceptibility

- [Parameters of Abstraction, Meaningfulness, and Pronunciability for 329 Nouns](https://www.sciencedirect.com/science/article/abs/pii/S0022537166800610?via%3Dihub)
- [Gated-Attention Readers for Text Comprehension](https://arxiv.org/abs/1606.01549)
- [Concreteness Ratings for 40 Thousand Generally Known English Word Lemmas](https://www.researchgate.net/publication/258061778_Concreteness_ratings_for_40_thousand_generally_known_English_word_lemmas)

#### Non-Specificity
- [WorNet Interface (Hypernyms)](https://www.nltk.org/howto/wordnet.html)
