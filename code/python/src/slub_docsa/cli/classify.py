"""Classify command.

Allows to train a classification model, persist it, and later classify new documents based on that model.
"""

# pylint: disable=no-member, too-many-locals, too-many-branches, too-many-statements

import argparse
import logging
import os
import pickle  # nosec

from slub_docsa.cli.common import add_logging_arguments, add_storage_directory_arguments, read_uft8_from_stdin
from slub_docsa.cli.common import setup_logging_from_args, setup_storage_directories
from slub_docsa.cli.qucosa import add_common_qucosa_arguments, available_qucosa_dataset_names
from slub_docsa.cli.qucosa import load_qucosa_dataset_by_name, load_qucosa_classification_model_by_name
from slub_docsa.cli.qucosa import available_qucosa_classification_model_names
from slub_docsa.common.dataset import SimpleDataset
from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.data.load.qucosa import read_qucosa_metadata_from_elasticsearch, _make_title_and_abstract_doc
from slub_docsa.data.load.qucosa import _make_title_only_doc, _make_title_and_fulltext_doc
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets, unique_subject_order

logger = logging.getLogger(__name__)


def classify_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _classify_qucosa_subparser(subparser.add_parser("qucosa"))


def _classify_qucosa_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify qucosa command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _classify_qucosa_train_subparser(subparser.add_parser("train"))
    _classify_qucosa_predict_subparser(subparser.add_parser("predict"))


def _classify_qucosa_train_action(args):
    """Perform training for single model and single qucosa dataset variant."""
    setup_logging_from_args(args)
    setup_storage_directories(args)

    dataset_name = args.dataset
    model_name = args.model
    persist_dir = args.persist_dir
    max_documents = args.limit
    check_qucosa_download = args.check_qucosa_download

    if persist_dir is None:
        persist_dir = os.path.join(get_cache_dir(), "models", dataset_name, model_name)

    # load dataset and model
    logger.info("load dataset '%s'", dataset_name)
    dataset, _ = load_qucosa_dataset_by_name(dataset_name, check_qucosa_download)

    if max_documents is not None:
        logger.info("use only the first %d examples for training", int(max_documents))
        documents = [dataset.documents[i] for i in range(int(max_documents))]
        subjects = [dataset.subjects[i] for i in range(int(max_documents))]
        dataset = SimpleDataset(documents, subjects)

    subject_order = unique_subject_order(dataset.subjects)
    subject_incidence = subject_incidence_matrix_from_targets(dataset.subjects, subject_order)
    logger.info("dataset '%s' has %d unique subjects", dataset_name, len(subject_order))

    logger.info("load model '%s'", model_name)
    model = load_qucosa_classification_model_by_name(model_name)

    if not isinstance(model, PersistableClassificationModel):
        logger.error("model '%s' is not persistable, abort", model_name)
        raise ValueError(f"model '{model_name}' is not persistable")

    logger.info("fit model '%s'", model_name)
    model.fit(dataset.documents, subject_incidence)

    logger.info("persist model '%s' to disk at '%s'", model_name, persist_dir)
    model.save(persist_dir)

    logger.info("persist subject order to disk at '%s'", persist_dir)
    with open(os.path.join(persist_dir, "subject_order.pickle"), "wb") as file:
        pickle.dump(subject_order, file)


def _classify_qucosa_predict_action(args):
    """Perform prediction for single qucosa dataset variant and model."""
    setup_logging_from_args(args)
    setup_storage_directories(args)

    dataset_name = args.dataset
    model_name = args.model
    persist_dir = args.persist_dir
    max_results = int(args.results)
    qucosa_id = args.id
    check_qucosa_download = args.check_qucosa_download

    if persist_dir is None:
        persist_dir = os.path.join(get_cache_dir(), "models", dataset_name, model_name)

    document = None
    if qucosa_id is not None:
        qucosa_metadata = next(iter(read_qucosa_metadata_from_elasticsearch(
            query={"terms": {"_id": [qucosa_id]}}
        )), None)
        if qucosa_metadata is None:
            raise ValueError(f"could not download/find qucosa document '{qucosa_id}'' on SLUB elastic search server")
        if "titles" in dataset_name:
            document = _make_title_only_doc(qucosa_metadata, "de")
            if document is None:
                raise ValueError("qucosa document does not provide german title, or title is too short")
        if "abstracts" in dataset_name:
            document = _make_title_and_abstract_doc(qucosa_metadata, "de")
            if document is None:
                raise ValueError("qucosa document does not provide german title and abstract")
        if "fulltexts" in dataset_name:
            document = _make_title_and_fulltext_doc(qucosa_metadata, "de")
            if document is None:
                raise ValueError("qucosa document does not provide german title and fulltext")
        logger.info("predict subjects for document: %s", str(document))
    else:
        text = read_uft8_from_stdin()
        if text is None:
            raise ValueError("you need to provide some text as input via stdin")
        document = Document(uri="stdin", title=None, fulltext=text)

    if document is None:
        raise ValueError("qucosa document not created succesfully, can't predict it")

    # load dataset and model
    logger.info("load dataset '%s'", dataset_name)
    _, subject_hierarchy = load_qucosa_dataset_by_name(dataset_name, check_qucosa_download)

    logger.info("load subject order from disk at '%s'", persist_dir)
    with open(os.path.join(persist_dir, "subject_order.pickle"), "rb") as file:
        subject_order = pickle.load(file)  # nosec

    logger.info("load model '%s'", model_name)
    model = load_qucosa_classification_model_by_name(model_name)

    if not isinstance(model, PersistableClassificationModel):
        logger.error("model '%s' is not persistable, abort", model_name)
        raise ValueError(f"model '{model_name}' is not persistable")

    # load persisted model and predict single document
    model.load(persist_dir)
    probabilities = model.predict_proba([document])[0]

    # prepare predictions for output
    predictions = list(reversed(sorted(zip(probabilities, subject_order))))
    if max_results > 0:
        predictions = predictions[:max_results]

    # print each prediction as one line to stdout
    for probability, subject_uri in predictions:
        if subject_hierarchy is not None and subject_uri in subject_hierarchy:
            label = subject_hierarchy.subject_labels(subject_uri) \
                .get("de").encode("ascii", errors="replace").decode("ascii")
            print(probability, subject_uri, label)
        else:
            print(probability, subject_uri)


def _add_classify_qucosa_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for classify qucosa train/predict commands."""
    add_storage_directory_arguments(parser)
    parser.add_argument(
        "--dataset",
        "-d",
        help="which dataset variants to use as training data: " + ", ".join(available_qucosa_dataset_names()),
        default="qucosa_de_fulltexts_langid_rvk",
    )

    parser.add_argument(
        "--model",
        "-m",
        help="which model variant to use for training: " + ", ".join(available_qucosa_classification_model_names(True)),
        default="tfidf_10k_torch_ann",
    )

    parser.add_argument(
        "--persist_dir",
        help="path to directory where model is saved",
    )


def _classify_qucosa_train_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify qucosa train command."""
    parser.set_defaults(func=_classify_qucosa_train_action)
    add_logging_arguments(parser)
    add_common_qucosa_arguments(parser)
    _add_classify_qucosa_common_arguments(parser)

    parser.add_argument(
        "--limit",
        "-l",
        help="""limit the number of training examples to this many examples,
        the default is that all examples are used for training""",
    )


def _classify_qucosa_predict_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify qucosa predict command."""
    parser.set_defaults(func=_classify_qucosa_predict_action)
    add_logging_arguments(parser)
    add_common_qucosa_arguments(parser)
    _add_classify_qucosa_common_arguments(parser)

    parser.add_argument(
        "--id",
        "-i",
        help="predict subjects for qucosa document with id instead of text from stdin",
    )

    parser.add_argument(
        "--results",
        "-r",
        help="the number of results, 0 means all, default is 10",
        default=10
    )
