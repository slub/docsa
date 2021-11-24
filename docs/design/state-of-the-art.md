# State of the Art: Bibliograhpic Classification

Contents
- [Scientific Publications](#scientific-publications)
  - Preprocessing & Feature Extraction
  - Multi-Label Text Classification
  - Hierarchical Classification
  - Bibliographic Classification
  - Evaluation Strategies
- [Existing Solutions](#existing-solutions)
- [Relevant Programming Libraries](#relevant-programming-libraries)

# Scientific Publications

## Preprocessing & Feature Extraction

### Fulltext Feature Extraction

General Preprocessing Methods:
- Lexical Analysis
- Stop Word Elimination
- Stemming / Lemmatiziation
- Vectorization

### Metadaten Feature Extraction

## Multi-label Text Classification

Approaches
- Classic methods
  - Instance-based methods (k-nearest neighbor search)
  - Tf-IdF vectorization + classification model (Support Vector Machines, Random Forests, etc.)
- ...
- Artificial Neural Networks
  - Convolutional Networks, see [Mullenbach et al.](https://aclanthology.org/N18-1100.pdf) (2018) and [Liu et al.](https://dl.acm.org/doi/pdf/10.1145/3077136.3080834) (2017)
  - BiGRU (Bi-directional Gated Recurrent) + Attention Networks, see [Chalkidis et al.](https://aclanthology.org/P19-1636.pdf) (2019) and [Xu et. al](http://proceedings.mlr.press/v37/xuc15.pdf) (2015)
  - BERT (Bi-directional Encoder Representations from Transformers), see [Chalkidis et al.](https://aclanthology.org/P19-1636.pdf) (2019), [code](https://github.com/iliaschalkidis/lmtc-eurlex57k)
  - X-BERT: eXtreme Multi-Label Text Classification with BERT by [Chang et al.](https://arxiv.org/pdf/1905.02331.pdf) (2019)
  - Overview Paper by [Minaee et al.](https://arxiv.org/pdf/2004.03705.pdf) (2021)

## Hierarchical Classification

General Approaches to Hierarchical Classification
- Flattening **vs.** Specialized Classification Algorithms
  - Flattening means to use standard classification approaches, but do special pre- and post-processing
    1. Use each class individually and assign training data based on parent/child classes
    2. Apply flat classification algorithms
    3. Analyse / reinterprete classification results
  - Specialized classifiers would know how to deal with hierachical information
- Individual models for each class or hierarchy level **vs.** one model for all classes

Related Overview Publications
- Stein, Roger Alan, Patricia A. Jaques, and Joao Francisco Valiati. "[An analysis of hierarchical text classification using word embeddings](https://arxiv.org/pdf/1809.01771)" (2019)

### Approach: Individial Models for each Class

- N. Cesa-Bianchi, C. Gentile, and L. Zaniboni. [Hierarchical classification: Combining Bayes with SVM](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7332&rep=rep1&type=pdf) (2006)

### Approach: One Model ~to rule them all~

### Approach: Models that understand hierarchical classes

- L. Cai and T. Hofmann. "[Hierarchical Document Categorization with Support Vector Machines](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.5940&rep=rep1&type=pdf)" (2004)

- W. Huang, et al. "[Hierarchical Multi-label Text Classification: An Attention-based
Recurrent Network Approach](https://bigdata.ustc.edu.cn/paper_pdf/2019/Wei-Huang-CIKM.pdf)" (2019)

- J. Risch, S. Garda, R. Krestel. "[Hierarchical Document Classification as a Sequence Generation Task](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020hierarchical.pdf)" (2020)

## Bibliographic Classification

Relevant Conferences
- [Semantic Web in Libraries](http://swib.org/) (SWIB)
- [European Library Automation Group](https://elag.org/) (ELAG)

Related Publications
- Golub, Koraljka, Johan Hagelbäck, and Anders Ardö. "[Automatic Classification of Swedish Metadata Using Dewey Decimal Classification: A Comparison of Approaches](https://www.sciendo.com/article/10.2478/jdis-2020-0003)" (2020)
- Lüschow, Andreas, and Christian Wartena. "[Classifying medical literature using k-nearest-neighbours algorithm](https://serwiss.bib.hs-hannover.de/frontdoor/deliver/index/docId/1146/file/Lueschow_Wartena_classifying.pdf)" (2017)

Critical Views
- Hjørland, Birger. "[Is classification necessary after Google?](https://sites.evergreen.edu/wp-content/uploads/sites/226/2016/08/hjorland-classification-after-google.pdf)." (2012)

## Evaluation Strategies

Main Strategies
- Direct assement of model performance by a human evaluator
- Automated comparison with a gold standard (split in training and test data)
- Indirectly via task-based performance

Hierarchical Score Functions
- H-Loss, assumes strong subtree annotation, see [Cesa-Bianchi et al.](https://www.jmlr.org/papers/volume7/cesa-bianchi06a/cesa-bianchi06a.pdf) (2006)
- Shortest Path, see [Cai et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.5940&rep=rep1&type=pdf)

Problems
- Balanced multi-label stratification, see [Sechidis et al.](https://link.springer.com/content/pdf/10.1007/978-3-642-23808-6_10.pdf)

# Existing Solutions

Open Source Projects
- [Annif](https://annif.org/) -
Tool for automated subject indexing and classification

Closed-Source Projects
- [JRC EuroVoc Indexer (JEX)](https://ec.europa.eu/jrc/en/language-technologies/jrc-eurovoc-indexer) for classifying [EUR-Lex](https://eur-lex.europa.eu/homepage.html?locale=de) documents according to the [EuroVoc](https://eur-lex.europa.eu/browse/eurovoc.html?locale=de) classes

# Relevant Programming Libraries

Bibliographic Classification
- [Annif](https://annif.org/) - Tool for automated subject indexing and classification

Preprocessing
- [nltk](https://www.nltk.org/) - Natural Language Toolkit
- [spacy](https://spacy.io/) - Natural Language Processing
- [gensim](https://radimrehurek.com/gensim/) - Topic Modelling / Vectorization / Similarity Search
- [sentence-transforms](https://github.com/UKPLab/sentence-transformers) Sentence Embeddings via Transformer Models

Deep Learning
- [ktrain](https://github.com/amaiya/ktrain)
- [huggingface](https://huggingface.co) - Many pre-trained deep learning models

General Machine Learning
- [scikit-learn](https://scikit-learn.org/) - Machine Learning in Python
- [scikit-multilearn](http://scikit.ml/) - Multi-Label Classification in Python
- [sklearn-hierarchical-classification](https://github.com/globality-corp/sklearn-hierarchical-classification) - Hierarchical classification algorithms

Research Papers with Code:
- [PapersWithCode](https://paperswithcode.com/dataset/glue)

# To Be Read

- Deep neural network for hierarchical extreme multi-label text
classification, [Gargiulo et al.](https://d1wqtxts1xzle7.cloudfront.net/63730011/Deep_neural_network_for_hierarchical_extreme_multi-with-cover-page-v2.pdf?Expires=1634292661&Signature=gjeQL8QOyP6C~4R1XMuJl98mWMYKNzxsZVo9YKOPJg8FNnUH3VGBNz9LYH6zLB5zae58itr-JcR6MYnOHThk4Rh3Tn2gMh1t0ZhSOwxJEdmbqcr6piV0OMZl5DXLoKb~Yra3lkT1VXKQCiwn~e9UXnDIX1qwSciA24HHXPxJ-uLCZMmio5zWdgh2dPlxr34mcxQouKvn-F0sqVtxz3i1i58bZbvnTjJWXhdhpRMCEyCCd4~BWOUqxXAkxcZjZnKeP1sUTptvu~vN1-SGOkgqnH61pv-302D4NLWCifd6l07cCeVRvo-Fv6C10OroL-ZWmypjImQzG6Kv1xU-KqZc9Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) (2019)