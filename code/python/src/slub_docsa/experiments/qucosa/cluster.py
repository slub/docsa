"""Clustering experiments for qucosa data."""

# pylint: disable=invalid-name

import os
import logging

import numpy as np

from sklearn.cluster import KMeans

from slub_docsa.common.paths import FIGURES_DIR, CACHE_DIR
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.experiments.common import get_qucosa_tfidf_stemming_vectorizer
from slub_docsa.experiments.qucosa.datasets import default_named_qucosa_datasets

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    random_state = 123
    plot_training_history_filepath = os.path.join(FIGURES_DIR, "qucosa/classify_one_ann_history")
    stemming_cache_filepath = os.path.join(CACHE_DIR, "stemming/global_cache.sqlite")
    _, dataset, _ = list(default_named_qucosa_datasets(["qucosa_de_fulltexts_langid_rvk"]))[0]

    vectorizer = get_qucosa_tfidf_stemming_vectorizer(max_features=10000)
    logger.debug("compile corpus")
    corpus = [document_as_concatenated_string(dataset.documents[i]) for i in range(1000)]
    logger.debug("fit vectorizer")
    vectorizer.fit(iter(corpus))
    logger.debug("vectorize documents")
    features = np.array(list(vectorizer.transform(iter(corpus))))
    logger.debug("features shape is %s", features.shape)

    cluster_model = KMeans()
    logger.debug("cluster fit")
    cluster_model.fit(features)
    logger.debug("cluster predict")
    assignments = cluster_model.predict(features)
    logger.debug("assignments are %s", str(assignments))
