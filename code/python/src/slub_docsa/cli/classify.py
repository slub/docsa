"""Classify command.

Allows to train a classification model, persist it, and later classify new documents based on that model.
"""

# pylint: disable=no-member, too-many-locals, too-many-branches, too-many-statements

import argparse
import logging
import os
import pickle  # nosec

from slub_docsa.cli.common import add_logging_arguments, add_storage_directory_arguments
from slub_docsa.cli.common import available_classification_model_names, read_uft8_from_stdin
from slub_docsa.cli.common import setup_logging_from_args, setup_storage_directories
from slub_docsa.cli.k10plus import available_k10plus_dataset_names, load_k10plus_dataset_by_name
from slub_docsa.cli.qucosa import add_common_qucosa_arguments, available_qucosa_dataset_names
from slub_docsa.cli.qucosa import load_qucosa_dataset_by_name
from slub_docsa.common.dataset import SimpleDataset
from slub_docsa.common.document import Document
from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.paths import get_cache_dir
from slub_docsa.data.load.qucosa import read_qucosa_metadata_from_elasticsearch, _make_title_and_abstract_doc
from slub_docsa.data.load.qucosa import _make_title_only_doc, _make_title_and_fulltext_doc
from slub_docsa.evaluation.classification.incidence import LazySubjectIncidenceTargets, unique_subject_order
from slub_docsa.serve.models.classification.common import get_all_classification_model_types

logger = logging.getLogger(__name__)


def _load_classification_model_by_name(model_name: str, subject_hierarchy, subject_order):
    """Return a single model retrieving it by its name."""
    model_types = get_all_classification_model_types()
    if model_name not in model_types:
        raise ValueError(f"model with name '{model_name}' not known")
    return model_types[model_name](subject_hierarchy, subject_order)


def classify_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for classify command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _train_predict_subparser(subparser.add_parser("qucosa"), "qucosa")
    _train_predict_subparser(subparser.add_parser("k10plus"), "k10plus")


def _train_predict_subparser(parser: argparse.ArgumentParser, datasource: str):
    """Return sub-parser for classify qucosa command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _classify_train_subparser(subparser.add_parser("train"), datasource)
    _classify_predict_subparser(subparser.add_parser("predict"), datasource)


def _classify_train_action_generator(datasource: str):
    """Return action that performs training for single model and single dataset variant."""

    def action(args):
        setup_logging_from_args(args)
        setup_storage_directories(args)

        dataset_name = args.dataset
        model_name = args.model
        persist_dir = args.persist_dir
        max_documents = args.limit
        check_qucosa_download = args.check_qucosa_download if datasource == "qucosa" else None

        if persist_dir is None:
            persist_dir = os.path.join(get_cache_dir(), "models", dataset_name, model_name)

        # load dataset and model
        logger.info("load dataset '%s'", dataset_name)
        if datasource == "qucosa":
            dataset, subject_hierarchy = load_qucosa_dataset_by_name(dataset_name, check_qucosa_download)
        elif datasource == "k10plus":
            dataset, subject_hierarchy = load_k10plus_dataset_by_name(dataset_name)
        else:
            raise ValueError(f"datasource '{datasource}' not supported")

        if max_documents is not None:
            logger.info("use only the first %d examples for training", int(max_documents))
            documents = [dataset.documents[i] for i in range(int(max_documents))]
            subjects = [dataset.subjects[i] for i in range(int(max_documents))]
            dataset = SimpleDataset(documents, subjects)

        subject_order = unique_subject_order(dataset.subjects)
        subject_incidence = LazySubjectIncidenceTargets(dataset.subjects, subject_order)
        logger.info("dataset '%s' has %d unique subjects", dataset_name, len(subject_order))

        logger.info("load model '%s'", model_name)
        model = _load_classification_model_by_name(model_name, subject_hierarchy, subject_order)

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

    return action


def _classify_predict_action_generator(datasource: str):
    """Return action that performs prediction for single dataset variant and model."""

    def action(args):
        setup_logging_from_args(args)
        setup_storage_directories(args)

        dataset_name = args.dataset
        model_name = args.model
        persist_dir = args.persist_dir
        max_results = int(args.results)
        qucosa_id = args.id
        check_qucosa_download = args.check_qucosa_download if datasource == "qucosa" else None

        if persist_dir is None:
            persist_dir = os.path.join(get_cache_dir(), "models", dataset_name, model_name)

        document = None
        if qucosa_id is not None:
            qucosa_metadata = next(iter(read_qucosa_metadata_from_elasticsearch(
                query={"terms": {"_id": [qucosa_id]}}
            )), None)
            if qucosa_metadata is None:
                raise ValueError(f"could not download qucosa document '{qucosa_id}'' on SLUB elastic search server")
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

            if document is None:
                raise ValueError("qucosa document not created succesfully, can't predict it")
            logger.info("predict subjects for document: %s", str(document))
        else:
            text = read_uft8_from_stdin()
            if text is None:
                raise ValueError("you need to provide some text as input via stdin")
            document = Document(uri="stdin", title=None, fulltext=text)

        # load dataset and model
        logger.info("load dataset '%s'", dataset_name)
        if datasource == "qucosa":
            _, subject_hierarchy = load_qucosa_dataset_by_name(dataset_name, check_qucosa_download)
        elif datasource == "k10plus":
            _, subject_hierarchy = load_k10plus_dataset_by_name(dataset_name)
        else:
            raise ValueError(f"datasource '{datasource}' not supported")

        logger.info("load subject order from disk at '%s'", persist_dir)
        with open(os.path.join(persist_dir, "subject_order.pickle"), "rb") as file:
            subject_order = pickle.load(file)  # nosec

        logger.info("load model '%s'", model_name)
        model = _load_classification_model_by_name(model_name, subject_hierarchy, subject_order)

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

    return action


def _add_classify_common_arguments(parser: argparse.ArgumentParser, datasource: str):
    """Add common arguments for classify qucosa train/predict commands."""
    available_datasets = []
    if datasource == "qucosa":
        available_datasets = available_qucosa_dataset_names()
    elif datasource == "k10plus":
        available_datasets = available_k10plus_dataset_names()

    add_storage_directory_arguments(parser)
    parser.add_argument(
        "--dataset",
        "-d",
        help="which dataset variants to use as training data: " + ", ".join(available_datasets),
        required=True
    )

    parser.add_argument(
        "--model",
        "-m",
        help="which model variant to use for training: " + ", ".join(available_classification_model_names()),
        required=True
    )

    parser.add_argument(
        "--persist_dir",
        help="path to directory where model is saved",
    )


def _classify_train_subparser(parser: argparse.ArgumentParser, datasource: str):
    """Return sub-parser for classify qucosa train command."""
    parser.set_defaults(func=_classify_train_action_generator(datasource))
    add_logging_arguments(parser)
    if datasource == "qucosa":
        add_common_qucosa_arguments(parser)
    _add_classify_common_arguments(parser, datasource)

    parser.add_argument(
        "--limit",
        "-l",
        help="""limit the number of training examples to this many examples,
        the default is that all examples are used for training""",
    )


def _classify_predict_subparser(parser: argparse.ArgumentParser, datasource: str):
    """Return sub-parser for classify qucosa predict command."""
    parser.set_defaults(func=_classify_predict_action_generator(datasource))
    add_logging_arguments(parser)
    if datasource == "qucosa":
        add_common_qucosa_arguments(parser)
    _add_classify_common_arguments(parser, datasource)

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
