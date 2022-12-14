"""Experiments command."""

import argparse
from slub_docsa.cli.common import add_logging_arguments, add_storage_directory_arguments, setup_logging_from_args
from slub_docsa.cli.qucosa import add_common_qucosa_arguments, available_qucosa_clustering_model_names
from slub_docsa.cli.qucosa import available_qucosa_classification_model_names, available_qucosa_dataset_names
from slub_docsa.cli.common import setup_storage_directories
from slub_docsa.experiments.qucosa.classify_many import qucosa_experiments_classify_many
from slub_docsa.experiments.qucosa.cluster_many import qucosa_experiments_cluster_many
from slub_docsa.experiments.qucosa.cluster_one import qucosa_experiments_cluster_one


def experiments_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for `experiments` command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _experiments_qucosa_subparser(subparser.add_parser("qucosa"))


def _experiments_qucosa_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for `experiments qucosa` command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _experiments_qucosa_classify_many_subparser(subparser.add_parser("classify_many"))
    _experiments_qucosa_cluster_many_subparser(subparser.add_parser("cluster_many"))
    _experiments_qucosa_cluster_one_subparser(subparser.add_parser("cluster_one"))


def _add_experiments_qucosa_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for experiments qucosa commands."""
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="a list of dataset variants to evaluate: " + ", ".join(available_qucosa_dataset_names()),
        default=["qucosa_de_titles_langid_rvk", "qucosa_de_abstracts_langid_rvk", "qucosa_de_fulltexts_langid_rvk"],
    )


def _experiments_qucosa_classify_many_action(args):
    """Perform `experiments qucosa classify_many` action."""
    setup_logging_from_args(args)
    setup_storage_directories(args)

    dataset_subset = args.datasets
    model_subset = args.models
    n_splits = int(args.cross_splits)
    check_qucosa_download = args.check_qucosa_download

    avaiable_datasets = available_qucosa_dataset_names()
    avaiable_models = available_qucosa_classification_model_names()

    for dataset_name in dataset_subset:
        if dataset_name not in avaiable_datasets:
            raise ValueError(f"dataset with name '{dataset_name}' is not available")

    for model_name in model_subset:
        if model_name not in avaiable_models:
            raise ValueError(f"model with name '{model_name}' is not available")

    qucosa_experiments_classify_many(
        dataset_subset=dataset_subset,
        model_subset=model_subset,
        n_splits=n_splits,
        random_state=None,
        load_cached_scores=False,
        split_function_name="random",
        stop_after_evaluating_split=None,
        check_qucosa_download=check_qucosa_download,
    )


def _experiments_qucosa_classify_many_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for `experiments qucosa classify_many` command."""
    parser.set_defaults(func=_experiments_qucosa_classify_many_action)
    add_logging_arguments(parser)
    add_storage_directory_arguments(parser)
    add_common_qucosa_arguments(parser)
    _add_experiments_qucosa_common_arguments(parser)

    model_names = available_qucosa_classification_model_names()
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="a list of classification models to evaluate: " + ", ".join(model_names),
        default=["nihilistic", "oracle", "tfidf_10k_knn_k=1"],
    )

    parser.add_argument(
        "--cross_splits",
        "-c",
        help="the number of cross-validation splits, default 10",
        default=10,
    )


def _experiments_qucosa_cluster_many_action(args):
    """Perform `experiments qucosa cluster_many` action."""
    setup_logging_from_args(args)
    setup_storage_directories(args)

    dataset_subset = args.datasets
    model_subset = args.models
    repeats = int(args.repeats)
    max_documents = None if args.limit is None else int(args.limit)
    check_qucosa_download = args.check_qucosa_download

    avaiable_datasets = available_qucosa_dataset_names()
    avaiable_models = available_qucosa_clustering_model_names()

    for dataset_name in dataset_subset:
        if dataset_name not in avaiable_datasets:
            raise ValueError(f"dataset with name '{dataset_name}' is not available")

    for model_name in model_subset:
        if model_name not in avaiable_models:
            raise ValueError(f"model with name '{model_name}' is not available")

    qucosa_experiments_cluster_many(
        dataset_subset=dataset_subset,
        model_subset=model_subset,
        repeats=repeats,
        max_documents=max_documents,
        check_qucosa_download=check_qucosa_download,
    )


def _experiments_qucosa_cluster_many_subparser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=_experiments_qucosa_cluster_many_action)
    add_logging_arguments(parser)
    add_storage_directory_arguments(parser)
    add_common_qucosa_arguments(parser)
    _add_experiments_qucosa_common_arguments(parser)

    model_names = available_qucosa_clustering_model_names()
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="a list of clustering models to evaluate: " + ", ".join(model_names),
        default=["random_c=20", "random_c=subjects", "tfidf_10k_kMeans_c=20", "tfidf_10k_kMeans_c=subjects"],
    )

    parser.add_argument(
        "--repeats",
        "-r",
        help="the number of clustering repetitons, default 10",
        default=10,
    )

    parser.add_argument(
        "--limit",
        "-l",
        help="the maximum number documents to consider for clustering, default all",
    )


def _experiments_qucosa_cluster_one_action(args):
    setup_logging_from_args(args)
    setup_storage_directories(args)

    dataset_name = args.dataset
    model_name = args.model
    max_documents = None if args.limit is None else int(args.limit)
    check_qucosa_download = args.check_qucosa_download

    qucosa_experiments_cluster_one(
        dataset_name,
        model_name,
        max_documents,
        check_qucosa_download
    )


def _experiments_qucosa_cluster_one_subparser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=_experiments_qucosa_cluster_one_action)
    add_logging_arguments(parser)
    add_common_qucosa_arguments(parser)
    add_storage_directory_arguments(parser)

    dataset_names = available_qucosa_dataset_names()
    model_names = available_qucosa_clustering_model_names()

    parser.add_argument(
        "--dataset",
        "-d",
        help="which dataset to cluster: " + ", ".join(dataset_names),
        default="qucosa_de_fulltexts_langid_rvk",
    )

    parser.add_argument(
        "--model",
        "-m",
        help="which clustering algorithm to apply: " + ", ".join(model_names),
        default="tfidf_10k_kMeans_c=20",
    )

    parser.add_argument(
        "--limit",
        "-l",
        help="the maximum number documents to consider for clustering, default all",
    )
