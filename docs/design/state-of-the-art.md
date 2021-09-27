# State of the Art: Bibliograhpic Classification

Contents
- [Scientific Publications](#scientific-publications)
  - Preprocessing & Feature Extraction
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

# Existing Solutions

Open Source Projects
- [Annif](https://annif.org/) -
Tool for automated subject indexing and classification

Closed-Source Projects
- [JRC EuroVoc Indexer (JEX)](https://ec.europa.eu/jrc/en/language-technologies/jrc-eurovoc-indexer) for classifying [EUR-Lex](https://eur-lex.europa.eu/homepage.html?locale=de) documents according to the [EuroVoc](https://eur-lex.europa.eu/browse/eurovoc.html?locale=de) classes


# Relevant Programming Libraries

Bibliographic Classification
- [Annif](https://annif.org/) -
Tool for automated subject indexing and classification

Preprocessing
- [nltk](https://www.nltk.org/) - Natural Language Toolkit
- [gensim](https://radimrehurek.com/gensim/) - Topic Modelling / Vectorization / Similarity Search

General Machine Learning
- [scikit-learn](https://scikit-learn.org/) - Machine Learning in Python
- [scikit-multilearn](http://scikit.ml/) - Multi-Label Classification in Python
- [sklearn-hierarchical-classification](https://github.com/globality-corp/sklearn-hierarchical-classification) - Hierarchical classification algorithms

# To Be Read

- [Large-Scale Multi-Label Text Classification on EU Legislation](https://arxiv.org/pdf/1906.02192v1.pdf) and [Extreme Multi-Label Legal Text Classification:
A case study in EU Legislation](https://arxiv.org/pdf/1905.10892.pdf)