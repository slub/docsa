"""Command Line Interface.

This library provides a basic command line interface to run common tasks, e.g., training and evaluating a
classification model for Qucosa documents.

# Available Commands

The following commands are implemented:

- **classify [qucosa|k10plus] train** - train a classification model using the Qucosa or k10plus datasets
- **classify [qucosa|k10plus] predict** - predict subjects for new documents based on a trained model
- **experiments [qucosa|k10plus] classify_many** - evaluate multiple classification models and generate plots
- **experiments qucosa cluster_many**- evaluate multiple clustering models and generate plots
- **experiments qucosa cluster_one** - evaluate a single clustering model and generate plots

Each command supports the following basic arguments:

- `-v` - increase the verbosity to debug messages
- `-s` - decrease the verbosity to warning and error messages
- `-h` - show a help message describing the command and its options

# Storage Configuration

All commands allow to support common arguments that allow to specify where resources, cached files and figures are
stored. The following directories are distinguished:

- `data_dir` - top-level directory for storing files
- `resources_dir` - directory used to stored downloaded but unprocessed resources
- `cache_dir` - directory used to to store various intermediate results
- `figures_dir` - the directory where plots and figures are generated
- `serve_dir` - the directory that contains models available for the REST service

All directories can be independently modified by command line arguments of the same name. By default, directories are
set to:

- `data_dir` is set as the **current working directory**
- `resources_dir` is `<data_dir>/resources`
- `cache_dir` is `<data_dir>/runtime/cache`
- `figures_dir` is `<data_dir>/runtime/figures`
- `serve_dir` is `<data_dir>/runtime/serve`

# Train a Classification Model and Predict

## Select a Dataset and Model

In order to select a Qucosa dataset variant as well as a specific classification model, the help message provides a
list of all available datasets and models:

```
slub_docsa classify qucosa train -h
```

The output will be:

```
usage: slub_docsa classify qucosa train [-h] [-v] [-s] [--data_dir DATA_DIR]
                                        [--resources_dir RESOURCES_DIR]
                                        [--cache_dir CACHE_DIR]
                                        [--figures_dir FIGURES_DIR]
                                        [--dataset DATASET] [--model MODEL]
                                        [--persist_dir PERSIST] [--limit LIMIT]
...
...
...
  --dataset DATASET, -d DATASET
                        which dataset variants to use as training data:
                        qucosa_all_titles_rvk, qucosa_all_titles_ddc,
                        qucosa_de_titles_rvk, qucosa_de_titles_ddc,
                        qucosa_de_titles_langid_rvk,
                        qucosa_de_titles_langid_ddc, qucosa_de_abstracts_rvk,
                        qucosa_de_abstracts_ddc,
                        qucosa_de_abstracts_langid_rvk,
                        qucosa_de_abstracts_langid_ddc,
                        qucosa_de_fulltexts_rvk, qucosa_de_fulltexts_ddc,
                        qucosa_de_fulltexts_langid_rvk,
                        qucosa_de_fulltexts_langid_ddc
  --model MODEL, -m MODEL
                        which model variant to use for training:
                        tfidf_2k_knn_k=1, tfidf_10k_knn_k=1,
                        tfidf_40k_knn_k=1, dbmdz_bert_sts1_knn_k=1,
                        dbmdz_bert_sts8_knn_k=1, random_vectorizer_knn_k=1,
                        tfidf_10k_knn_k=3, tfidf_10k_dtree, tfidf_10k_rforest,
                        dbmdz_bert_sts1_rforest, tfidf_10k_scikit_mlp,
                        tfidf_2k_torch_ann, tfidf_10k_torch_ann,
                        tfidf_40k_torch_ann, dbmdz_bert_sts1_scikit_mlp,
                        dbmdz_bert_sts1_torch_ann, dbmdz_bert_sts8_torch_ann,
                        tfidf_10k_log_reg, tfidf_10k_nbayes, tfidf_10k_svc
...
```

Note that not all models are supported in combination with this command, since not all models can be saved to disk for
later re-use, e.g., the `slub_docsa.models.classification.dummy.OracleModel` will not work here.

Qucosa dataset variants are distinguished by:

- `all` vs `de` - whether documents are filtered based on their provided language information (`all` means no
  filtering, `de` means only german documents)
- `titles`, `abstracts`, `fulltexts` - what information is compiled into a document for training (titles only, titles +
  abstracts, titles + fulltexts)
- `langid` - whether the document language is further checked and filtered using the `langid` python module
- `rvk` vs `ddc` - which subject annotations are extracted for the documents and used for training and prediction

Model variants are distinguished by:

- `tfidf` vs `dbmdz_bert` - which vectorization strategy is applied, see
  `slub_docsa.data.preprocess.vectorizer.GensimTfidfVectorizer` and
  `slub_docsa.data.preprocess.vectorizer.HuggingfaceBertVectorizer`
- `2k`, `10k`, `40k` - the size of tfidf vectors that are being generated
- `sts1` vs `sts8` - the number of sub-text samples that are being vectorized using the DBMDZ pre-trained Bert model
- `knn`, `rforest`, `dtree`, `torch_ann` - which machine learning model is used for training

## Train a Model

Let's train a model using a the Qucosa data variant `qucosa_de_titles_langid_rvk` and the classification model
`tfidf_10k_torch_ann`.

```
slub_docsa classify qucosa train \\
    -d qucosa_de_titles_langid_rvk \\
    -m tfidf_10k_torch_ann
```

A number of task are performed:

1. The Qucosa data is being downloaded (if not present already) from the SLUB Elasticsearch server. Make sure to supply
   the username and password via environment variables `SLUB_ELASTICSEARCH_SERVER_USER` and
   `SLUB_ELASTICSEARCH_SERVER_PASSWORD`, see the module `slub_docsa.data.load.qucosa`.
2. Qucosa documents are parsed, checked for their language, and stored in a database file for easy access later on in
   the `<cache_dir>`.
3. Vectors are generated for each document depending on the chosen model and cached in the `<cache_dir>`.
4. The machine learning training algorithm is run with all available Qucosa documents.
5. The vectorization model and classification model are stored in `<persist_dir>`, by default in
   `<data_dir>/models/<dataset>/<model>/`.

Note that:

- If pre-processing of the data fails at some point, a number of files might have been generated in `<cache_dir>`.
  A subsequent run might try to load data from these incomplete cached files. Therefore, if in doubt, delete
  `<cache_dir>` until all files are succesfully pre-processed without any problems.

## Predict based on Arbitrary Text

Arbitrary text can be classified by providing it as input to the command:

```
echo "Verkehrsplanung" | slub_docsa classify qucosa predict \\
    -d qucosa_de_titles_langid_rvk \\
    -m tfidf_10k_torch_ann
```

The output will be a lot of log messages reporting the progress (which can be silenced via `-s`) and the final
prediction results consisting of the score, the subject URI and its label (if available):

```
0.9997003 https://rvk.uni-regensburg.de/api/xml/node/QR%20800 Allgemeines
0.9934012 https://rvk.uni-regensburg.de/api/xml/node/ZO%203300 Verkehrsmittel;  Verkehrsmittelwahl
0.02613362 https://rvk.uni-regensburg.de/api/xml/node/QR%20800%20-%20QR%20870 Verkehrswesen
0.0009589965 https://rvk.uni-regensburg.de/api/xml/node/ZO%204620 Verkehrsleitsysteme, Telematik im Stra?enverkehr
0.00043443392 https://rvk.uni-regensburg.de/api/xml/node/QR%20860 Stadtverkehrsprobleme
0.00036961303 https://rvk.uni-regensburg.de/api/xml/node/MS Spezielle Soziologien
0.00018879405 https://rvk.uni-regensburg.de/api/xml/node/AR%2022240 Wasserhygiene, Gew?sserschutz, Gew?ssersanierung
0.00012006318 https://rvk.uni-regensburg.de/api/xml/node/NZ%2013380 Einzelne Pers?nlichkeiten (alph.)
9.0069545e-05 https://rvk.uni-regensburg.de/api/xml/node/QR%20760 Neue Medien. Online-Dienste (Internet u. a.)
8.523816e-05 https://rvk.uni-regensburg.de/api/xml/node/WQ%203600 Pterygota (Fluginsekten), Palaeoptera (Altfl?gler), \
Neoptera (Neufl?gler), Polyneoptera
```

## Predict based on A New Qucosa Document

A new Qucosa document can be classified by referencing its id, e.g., `oai:qucosa:de:qucosa:21723` using the `-i`
parameter:

```
slub_docsa classify qucosa predict \\
    -i "oai:qucosa:de:qucosa:74587" \\
    -d qucosa_de_titles_langid_rvk \\
    -m tfidf_10k_torch_ann
```

Similar to before, the output will consist of a sorted list of subject predictions:

```
0.9923579 https://rvk.uni-regensburg.de/api/xml/node/ZO%204000%20-%20ZO%204999 Stra?enverkehr
0.0031882976 https://rvk.uni-regensburg.de/api/xml/node/ZG%209148 Technische Formgebung (Industrial Design)
2.399958e-05 https://rvk.uni-regensburg.de/api/xml/node/DO Spezialfragen des gesamten Schulsystems
1.9707892e-05 https://rvk.uni-regensburg.de/api/xml/node/ZC%2055000%20-%20ZC%2056900 Obstarten
6.9052834e-07 https://rvk.uni-regensburg.de/api/xml/node/DK%201061 Sachsen
6.7059597e-07 https://rvk.uni-regensburg.de/api/xml/node/ZO%204875 Unfallforschung im Stra?enverkehr
6.6172976e-07 https://rvk.uni-regensburg.de/api/xml/node/ZO%204340 Fahrr?der
4.6475782e-07 https://rvk.uni-regensburg.de/api/xml/node/QP%20500%20-%20QP%20550 Produktion. Beschaffung und \
Lagerhaltung
3.3459614e-07 https://rvk.uni-regensburg.de/api/xml/node/AL%2047000%20-%20AL%2048450 Soziales und Freizeit
1.4508818e-07 https://rvk.uni-regensburg.de/api/xml/node/QA%2010000 Zeitschriften
```

# Evaluate and Compare Multiple Models and Datasets

In order to better understand how well different models perform in different scenarios, the following commands allow to
compare various performance scores for multiple combinations of models and dataset variants.

## Classification Experiments

Classification models are evaluated using the
[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) strategy. Each dataset in split into
a training and test set. Models are rated based on how good they are able to predict subjects for the test set, which
was not used during training. A number of scores reflect different perspectives on their predictive performance.

Use `-h` to print a list of all available models and datasets:

```
slub_docsa experiments qucosa classify_many -h
```

The following command evaluates three models and three Qucosa dataset variants using a 10-fold cross-validation
strategy and saves various plots illustrating results in the directory `<figures_dir>/qucosa`.

```
slub_docsa experiments qucosa classify_many \\
    -d qucosa_de_titles_langid_rvk qucosa_de_abstracts_langid_rvk qucosa_de_fulltexts_langid_rvk \\
    -m nihilistic oracle tfidf_10k_knn_k=1
    -c 10
```

Note: This command will usually take a long time, since many processing steps need to be repeated for different models
and cross-validation splits.

The following plots will be generated:

- `slub_docsa.evaluation.plotting.score_matrices_box_plot` - performance scores for all models and dataset variants;
  score variance over multiple cross-validation splits is visualized as a box plot
- `slub_docsa.evaluation.plotting.score_matrix_box_plot` - performance scores for each model and a single dataset
  variant; score variance over multiple cross-validation splits is visualized as box plot
- `slub_docsa.evaluation.plotting.precision_recall_plot` - illustrates how models balance precision and recall
- `slub_docsa.evaluation.plotting.per_subject_precision_recall_vs_samples_plot` - per subject precision and recall
  scores in comparison to the number of availabe test examples; illustrates that subjects with insufficient number of
  training and test examples are usually more difficult to predict
- `slub_docsa.evaluation.plotting.per_subject_score_histograms_plot` - per subject score histograms, which illustrates
  how many subjects can be predicted with high/low precision and recall

For performance scoring, a number of methods are used. Some scores are based on predicted subject probabilities. Others
require a crisp decisions (yes/no) for each subject. Since there are multiple strategies on how to come up with
decisions based on subject probabilities, two common methods are distinguished:

- `t=best` - a threshold is estimated which decides when a subject is predicted (if its probability is larger than the
    threshold)
- `top3` - only the 3 subjects with highest probabilities are predicted (unless there are less than 3 subjects with
  probabilities larger than 0)

Based on these crisp subject predictions, the following scores are calculated:

- `precision micro`, `recall micro`, `f1_score micro` - see
  [precision and recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- `h_loss` - a loss function that considers hierarchical relations between subjects, see
  `slub_docsa.evaluation.classification.score.hierarchical`

In addition, the following scores are calculated based on subject probabilties:

- `roc auc micro` - the
  [receiver operating characteristic](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) (roc)
- `log loss` - the [cross-entropy loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)


## Clustering Experiments

Clustering models are either evaluated in comparison to each other, or one at a time.

```
slub_docsa experiments qucosa cluster_many -h
```

```
usage: slub_docsa experiments qucosa cluster_many [-h] [-v] [-s]
                                                  [--datasets DATASETS [DATASETS ...]]
                                                  [--models MODELS [MODELS ...]]
                                                  [--repeats REPEATS]
                                                  [--limit LIMIT]
...
  --models MODELS [MODELS ...], -m MODELS [MODELS ...]
                        a list of clustering models to evaluate: random_c=20,
                        random_c=subjects, tfidf_10k_kMeans_c=20,
                        tfidf_10k_kMeans_c=subjects,
                        tfidf_10k_agg_sl_cosine_c=subjects,
                        tfidf_10k_agg_sl_eucl_c=subjects,
                        tfidf_10k_agg_ward_eucl_c=subjects, bert_kMeans_c=20,
                        bert_kMeans_c=subjects,
                        bert_kMeans_agg_sl_eucl_c=subjects,
                        bert_kMeans_agg_ward_eucl_c=subjects
```

```
slub_docsa experiments qucosa cluster_many \\
    -d qucosa_de_titles_langid_rvk qucosa_de_abstracts_langid_rvk qucosa_de_fulltexts_langid_rvk \\
    -m random_c=20 tfidf_10k_kMeans_c=20 -r 1
```

Resulting plots is `slub_docsa.evaluation.plotting.score_matrices_box_plot` with several scores, see
[Clustering Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation):

- `mutual info` - comparing clusters to true labels by measuring the degree of agreement betwen the two assignments
- `rand` - a measure calculated by counting how many cluster assignments match true labels
- `homogeneity` and `completeness` - measuring whether a cluster only contains members of a single subject vs. whether
  all documents of a subject are assigned to the same cluster
- `intra cluster tfidf cosine` - measures the average distance between two documents of the same cluster


In order to get a better understanding of how well clusters match true subjects, the following commands allows to
generate interactive plots that allow to explore which documents are assigned to which clusters and are associated with
certain subjects:

```
slub_docsa experiments qucosa cluster_one \\
    -d qucosa_de_titles_langid_rvk \\
    -m tfidf_10k_kMeans_c=20
```

The resulting plots are:

- `slub_docsa.evaluation.plotting.subject_distribution_by_cluster_plot`
- `slub_docsa.evaluation.plotting.cluster_distribution_by_subject_plot`

"""
