"""Command Line Interface.

This library provides a basic command line interface to run common tasks, e.g., training and evaluating a
classification model for Qucosa documents.

## Available Commands

The following commands are implemented:

- **classify qucosa train** - train a classification model using the Qucosa dataset
- **classify qucosa predict** - predict subjects for new documents based on a trained model
- **experiments qucosa classify_many** - evaluate multiple classification models and generate plots
- **experiments qucosa cluster_many**- evaluate multiple clustering models and generate plots
- **experiments qucosa cluster_one** - evaluate a single clustering model and generate plots

Each command supports the following basic arguments:

- `-v` - increase the verbosity to debug messages
- `-s` - decrease the verbosity to warning and error messages
- `-h` - show a help message describing the command and its options

## Storage Configuration

All commands allow to support common arguments that allow to specify where resources, cached files and figures are
stored. The following directories are distinguished:

- `data_dir` - top-level directory for storing files
- `resources_dir` - directory used to stored downloaded but unprocessed resources
- `cache_dir` - directory used to to store various intermediate results
- `figures_dir` - the directory where plots and figures are generated

All directories can be independently modified by command line arguments of the same name. By default, directories are
set to:

- `data_dir` is set as the current working directory
- `resources_dir` is `<data_dir>/resources`
- `cache_dir` is `<data_dir>/runtime/cache`
- `figures_dir` is `<data_dir/runtime/figures`

## Train a Classification Model and Predict

### Select a Dataset and Model

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

Note that not all models supported by this library are available since not all models can be saved to disk for later
re-use, e.g., the `slub_docsa.models.classification.dummy.OracleModel` is not listed here. Also, none of the Annif
models are supported by this command due to technical issues. However, Annif itself provides a similar command line
interface that can be used for the same purpose.

Datasets variants are distinguished by:

- `all` vs `de` - whether documents are filtered based on their provided language information (`all` means no
  filtering, `de` means only german documents)
- `titles`, `abstracts`, `fulltexts` - what information is compiled into a document for training (titles only, titles +
  abstracts, titles + fulltexts)
- `langid` - whether the document language is further checked and filtered using the `langid` python module
- `rvk` vs `ddc` - which subject annotations are extracted for the documents and used for training and prediction

Model variants are distinguished by:

- `tfidf` vs `dbmdz_bert` - which vectorization strategy is applied, see
  `slub_docsa.data.preprocess.vectorizer.TfidfStemmingVectorizer` and
  `slub_docsa.data.preprocess.vectorizer.HuggingfaceBertVectorizer`
- `2k`, `10k`, `40k` - the size of tfidf vectors that are being generated
- `sts1` vs `sts8` - the number of sub-text samples that are being vectorized using the DBMDZ pre-trained Bert model
- `knn`, `rforest`, `dtree`, `torch_ann` - which machine learning model is used for training

### Train a Model

Let's train a model using a the Qucosa data variant `qucosa_de_fulltexts_langid_rvk` and the classification model
`dbmdz_bert_sts1_torch_ann`.

```
slub_docsa classify qucosa train \\
    -d qucosa_de_fulltexts_langid_rvk \\
    -m dbmdz_bert_sts1_torch_ann
```

A number of task are performed:

1. The Qucosa data is being downloaded (if not present already) from the SLUB Elasticsearch server. Make sure to supply
   the username and password via environment variables `SLUB_ELASTICSEARCH_SERVER_USER` and
   `SLUB_ELASTICSEARCH_SERVER_PASSWORD`, see the module `slub_docsa.data.load.qucosa`.
2. Qucosa documents are parsed, checked for their language, and stored in as database file for easy access later on in
   the `<cache_dir>`.
3. Vectors are generated for each document depending on the chosen model and cached in the `<cache_dir>`.
4. The machine learning training algorithm is run with all available Qucosa documents.
5. The vectorization model and classification model are stored in `<persist_dir>`, by default in
   `<data_dir>/models/<dataset>/<model>/`.

Note that:

- If pre-processing of the data fails as some point, a number of files might have been downloaded in `<resources_dir>`
  and generated in `<cache_dir>`. A subsequent run might try to load data from these incomplete downloads and
  incomplete cached files. Therefore, if in doubt, delete both `<resources_dir>` and `<cache_dir>` until all files are
  succesfully downloaded and pre-processed without any problems.

### Predict based on Arbitrary Text

Arbitrary text can be classified by providing it as input to the command:

```
echo "Some text that is being classified" | slub_docsa classify qucosa predict \\
    -d qucosa_de_fulltexts_langid_rvk \\
    -m dbmdz_bert_sts1_torch_ann
```

The output will be a lot of log messages reporting the progress (which can be silenced via `-s`) and the final
prediction results consisting of the score, the subject URI and its label (if available):

```
TODO TODO TODO
```

### Predict based on A New Qucosa Document

A new Qucosa document can be classified by referencing its id, e.g., `oai:qucosa:de:qucosa:21723` using the `-i`
parameter:

```
slub_docsa classify qucosa predict \\
    -i "oai:qucosa:de:qucosa:21723" \\
    -m dbmdz_bert_sts1_torch_ann \\
    -d qucosa_de_fulltexts_langid_rvk
```

Similar to before, the output will consist of a sorted list of subject predictions:

```
0.7393395 https://rvk.uni-regensburg.de/api/xml/node/UF Mechanik, Kontinuumsmechanik, Str?mungsmechanik, \
Schwingungslehre
0.09655364 https://rvk.uni-regensburg.de/api/xml/node/ZI%207100 Massivbau
0.018514823 https://rvk.uni-regensburg.de/api/xml/node/ST%20320%20-%20ST%20690 Einzelne Anwendungen der \
Datenverarbeitung
0.0025747041 https://rvk.uni-regensburg.de/api/xml/node/ZM%203500%20-%20ZM%203900 Werkstoffpr?fung allgemein
0.0015783093 https://rvk.uni-regensburg.de/api/xml/node/ZI%204000%20-%20ZI%204960 Bauphysik, Baustoffe, Bausch?den
0.0013172389 https://rvk.uni-regensburg.de/api/xml/node/UP%207500 Allgemeines
0.00043334285 https://rvk.uni-regensburg.de/api/xml/node/ZI%204500%20-%20ZI%204530 Beton
0.00021226388 https://rvk.uni-regensburg.de/api/xml/node/ZL%204100%20-%20ZL%204180 Getriebetechnik, Getriebe
0.0001846007 https://rvk.uni-regensburg.de/api/xml/node/ZS Handwerk, Feinwerktechnik, Holztechnik, Papiertechnik, \
Textiltechnik und sonstige Technik
0.00015280316 https://rvk.uni-regensburg.de/api/xml/node/ZM%207020%20-%20ZM%207029 Verbundwerkstoffe, Faserverst?rkte \
Werkstoffe
```

## Evaluate and Compare Multiple Models and Datasets

In order to better understand how well different models perform in different scenarios, the following commands allow to
compare various scores for multiple combinations of models and dataset variants.

### Classification Experiments

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
  score variance over multiple cross-validation splits is visualized as a box
- `slub_docsa.evaluation.plotting.score_matrix_box_plot` - performance scores for each model and a single dataset
  variant; score variance over multiple cross-validation splits is visualized as box
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
  `slub_docsa.evaluation.score.cesa_bianchi_h_loss`

In addition, the following scores are calculated based on subject probabilties:

- `roc auc micro` - the
  [receiver operating characteristic](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) (roc)
- `log loss` - the [cross-entropy loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)


### Clustering Experiments



"""
