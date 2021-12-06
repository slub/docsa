"""Classify command.

Allows to train a classification model, persist it, and later classify new documents based on that model.
"""

# pylint: disable=no-member, too-many-locals

import argparse
import logging
import os
import pickle  # nosec

from slub_docsa.cli.common import add_logging_arguments, read_uft8_from_stdin, setup_logging_from_args
from slub_docsa.cli.qucosa import available_qucosa_dataset_names, available_qucosa_model_names
from slub_docsa.cli.qucosa import load_qucosa_dataset_by_name, load_qucosa_model_by_name
from slub_docsa.common.dataset import SimpleDataset
from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel
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

    dataset_name = args.dataset
    model_name = args.model
    persist_path = args.persist

    # load dataset and model
    logger.info("load dataset '%s'", dataset_name)
    dataset, _ = load_qucosa_dataset_by_name(dataset_name)

    documents = [dataset.documents[i] for i in range(1000)]
    subjects = [dataset.subjects[i] for i in range(1000)]

    dataset = SimpleDataset(documents, subjects)

    subject_order = unique_subject_order(dataset.subjects)
    subject_incidence = subject_incidence_matrix_from_targets(dataset.subjects, subject_order)
    logger.info("dataset '%s' has %d unique subjects", dataset_name, len(subject_order))

    logger.info("load model '%s'", model_name)
    model = load_qucosa_model_by_name(model_name)

    if not isinstance(model, PersistableClassificationModel):
        logger.error("model '%s' is not persistable, abort", model_name)
        raise ValueError(f"model '{model_name}' is not persistable")

    logger.info("fit model '%s'", model_name)
    model.fit(dataset.documents, subject_incidence)

    logger.info("persist model '%s' to disk at '%s'", model_name, persist_path)
    model.save(persist_path)

    logger.info("persist subject order to disk at '%s'", persist_path)
    with open(os.path.join(persist_path, "subject_order.pickle"), "wb") as file:
        pickle.dump(subject_order, file)


def _classify_qucosa_predict_action(args):
    """Perform prediction for single qucosa dataset variant and model."""
    setup_logging_from_args(args)

    dataset_name = args.dataset
    model_name = args.model
    persist_path = args.persist
    max_results = int(args.results)
    text = read_uft8_from_stdin()

    document = Document(uri="stdin", title=None, fulltext=text)

    # load dataset and model
    logger.info("load dataset '%s'", dataset_name)
    _, subject_hierarchy = load_qucosa_dataset_by_name(dataset_name)

    logger.info("load subject order from disk at '%s'", persist_path)
    with open(os.path.join(persist_path, "subject_order.pickle"), "rb") as file:
        subject_order = pickle.load(file)  # nosec

    logger.info("load model '%s'", model_name)
    model = load_qucosa_model_by_name(model_name)

    if not isinstance(model, PersistableClassificationModel):
        logger.error("model '%s' is not persistable, abort", model_name)
        raise ValueError(f"model '{model_name}' is not persistable")

    # load persisted model and predict single document
    model.load(persist_path)
    probabilities = model.predict_proba([document])[0]

    # prepare predictions for output
    predictions = list(reversed(sorted(zip(probabilities, subject_order))))
    if max_results > 0:
        predictions = predictions[:max_results]

    # print each prediction as one line to stdout
    for probability, subject_uri in predictions:
        if subject_hierarchy is not None and subject_uri in subject_hierarchy:
            label = subject_hierarchy[subject_uri].label.encode("ascii", errors="replace").decode("ascii")
            print(probability, subject_uri, label)
        else:
            print(probability, subject_uri)


def _add_classify_qucosa_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for classify qucosa train/predict commands."""
    parser.add_argument(
        "--dataset",
        "-d",
        help="which dataset variants to use as training data: " + ", ".join(available_qucosa_dataset_names()),
        default="qucosa_de_fulltexts_langid_rvk",
    )

    parser.add_argument(
        "--model",
        "-m",
        help="which model variant to use for training: " + ", ".join(available_qucosa_model_names(True)),
        default="tfidf 10k torch ann",
    )

    parser.add_argument(
        "--persist",
        "-p",
        help="path to directory where model is saved",
        required=True
    )


def _classify_qucosa_train_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify qucosa train command."""
    parser.set_defaults(func=_classify_qucosa_train_action)
    add_logging_arguments(parser)
    _add_classify_qucosa_common_arguments(parser)


def _classify_qucosa_predict_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify qucosa predict command."""
    parser.set_defaults(func=_classify_qucosa_predict_action)
    add_logging_arguments(parser)
    _add_classify_qucosa_common_arguments(parser)

    parser.add_argument(
        "--results",
        "-r",
        help="the number of results, 0 means all",
        default=10
    )
