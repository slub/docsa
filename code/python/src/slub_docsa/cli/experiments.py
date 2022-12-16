"""Experiments command."""

import argparse
from slub_docsa.cli.common import add_logging_arguments, add_storage_directory_arguments
from slub_docsa.cli.common import available_classification_model_names, setup_logging_from_args
from slub_docsa.cli.common import available_dataset_names_by_datasource
from slub_docsa.cli.k10plus import available_k10plus_dataset_names
from slub_docsa.cli.qucosa import add_common_qucosa_arguments, available_qucosa_clustering_model_names
from slub_docsa.cli.qucosa import available_qucosa_dataset_names
from slub_docsa.cli.common import setup_storage_directories
from slub_docsa.experiments.k10plus.classify_many import k10plus_experiments_classify_many
from slub_docsa.experiments.qucosa.classify_many import qucosa_experiments_classify_many
from slub_docsa.experiments.qucosa.cluster_many import qucosa_experiments_cluster_many
from slub_docsa.experiments.qucosa.cluster_one import qucosa_experiments_cluster_one


def experiments_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for `experiments` command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _experiments_method_subparser(subparser.add_parser("qucosa"), "qucosa")
    _experiments_method_subparser(subparser.add_parser("k10plus"), "k10plus")


def _experiments_method_subparser(parser: argparse.ArgumentParser, datasource: str):
    """Return sub-parser for `experiments qucosa` command."""
    parser.set_defaults(func=lambda args: parser.print_help())
    subparser = parser.add_subparsers()
    _experiments_classify_many_subparser(subparser.add_parser("classify_many"), datasource)
    _experiments_cluster_many_subparser(subparser.add_parser("cluster_many"), datasource)
    _experiments_cluster_one_subparser(subparser.add_parser("cluster_one"), datasource)


def _check_dataset_subset_is_valid(dataset_subset, datasource):
    available_datasets = available_dataset_names_by_datasource(datasource)

    for dataset_name in dataset_subset:
        if dataset_name not in available_datasets:
            raise ValueError(f"dataset with name '{dataset_name}' is not available")


def _check_clustering_model_subset_is_valid(model_subset):
    available_models = available_qucosa_clustering_model_names()
    for model_name in model_subset:
        if model_name not in available_models:
            raise ValueError(f"model with name '{model_name}' is not available")


def _check_classification_model_subset_is_valid(model_subset):
    available_models = available_classification_model_names()
    print(model_subset)
    for model_name in model_subset:
        if model_name not in available_models:
            raise ValueError(f"model with name '{model_name}' is not available")


def _add_experiments_common_arguments(parser: argparse.ArgumentParser, datasource: str):
    """Add common arguments for experiments qucosa commands."""
    available_datasets = []
    if datasource == "qucosa":
        available_datasets = available_qucosa_dataset_names()
    elif datasource == "k10plus":
        available_datasets = available_k10plus_dataset_names()
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="a list of dataset variants to evaluate: " + ", ".join(available_datasets),
        required=True,
    )


def _experiments_classify_many_action_generator(datasource: str):
    """Return action method that performs classification models comparison experiments."""

    def action(args):
        setup_logging_from_args(args)
        setup_storage_directories(args)

        dataset_subset = args.datasets
        model_subset = args.models
        n_splits = int(args.cross_splits)
        cross_stop = n_splits if args.cross_stop is None else int(args.cross_stop)
        load_cached_scores = args.load_cached_scores
        check_qucosa_download = args.check_qucosa_download if datasource == "qucosa" else None

        _check_dataset_subset_is_valid(dataset_subset, datasource)
        _check_classification_model_subset_is_valid(model_subset)

        if datasource == "qucosa":
            qucosa_experiments_classify_many(
                dataset_subset=dataset_subset,
                model_subset=model_subset,
                n_splits=n_splits,
                random_state=None,
                load_cached_scores=load_cached_scores,
                split_function_name="random",
                stop_after_evaluating_split=cross_stop,
                check_qucosa_download=check_qucosa_download,
            )
        if datasource == "k10plus":
            k10plus_experiments_classify_many(
                dataset_subset=dataset_subset,
                model_subset=model_subset,
                n_splits=n_splits,
                random_state=None,
                load_cached_scores=load_cached_scores,
                split_function_name="random",
                stop_after_evaluating_split=cross_stop,
            )

    return action


def _experiments_classify_many_subparser(parser: argparse.ArgumentParser, datasource: str):
    """Return sub-parser for `experiments qucosa classify_many` command."""
    parser.set_defaults(func=_experiments_classify_many_action_generator(datasource))
    add_logging_arguments(parser)
    add_storage_directory_arguments(parser)
    if datasource == "qucosa":
        add_common_qucosa_arguments(parser)
    _add_experiments_common_arguments(parser, datasource)

    model_names = available_classification_model_names()
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="a list of classification models to evaluate: " + ", ".join(model_names),
        required=True,
    )

    parser.add_argument(
        "--cross_splits",
        "-c",
        help="the number of cross-validation splits, default 10",
        default=10,
    )

    parser.add_argument(
        "--cross_stop",
        "-x",
        type=int,
        help="the number of the cross-validation split at which to stop prematurely, default none",
        default=None
    )

    parser.add_argument(
        "--load_cached_scores",
        "-l",
        action="store_true",
        help="whether to load already calculated scores from the cache, e.g., if a prior run was interrupted",
        default=False
    )


def _experiments_cluster_many_action_generator(datasource: str):
    """Return action that performs clustering experiments for multiple datasets."""

    def action(args):
        setup_logging_from_args(args)
        setup_storage_directories(args)

        dataset_subset = args.datasets
        model_subset = args.models
        repeats = int(args.repeats)
        max_documents = None if args.limit is None else int(args.limit)
        check_qucosa_download = args.check_qucosa_download if datasource == "qucosa" else None

        _check_dataset_subset_is_valid(dataset_subset, datasource)
        _check_clustering_model_subset_is_valid(model_subset)

        if datasource == "qucosa":
            qucosa_experiments_cluster_many(
                dataset_subset=dataset_subset,
                model_subset=model_subset,
                repeats=repeats,
                max_documents=max_documents,
                check_qucosa_download=check_qucosa_download,
            )
        else:
            raise ValueError(f"datasource '{datasource}' not supported for clustering")

    return action


def _experiments_cluster_many_subparser(parser: argparse.ArgumentParser, datasource: str):
    parser.set_defaults(func=_experiments_cluster_many_action_generator(datasource))
    add_logging_arguments(parser)
    add_storage_directory_arguments(parser)
    add_common_qucosa_arguments(parser)
    _add_experiments_common_arguments(parser, datasource)

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


def _experiments_cluster_one_action_generator(datasource: str):
    """Return method that performs clustering one a single dataset."""

    def action(args):
        setup_logging_from_args(args)
        setup_storage_directories(args)

        dataset_name = args.dataset
        model_name = args.model
        max_documents = None if args.limit is None else int(args.limit)
        check_qucosa_download = args.check_qucosa_download

        _check_dataset_subset_is_valid([dataset_name], datasource)
        _check_clustering_model_subset_is_valid([model_name])

        if datasource == "qucosa":
            qucosa_experiments_cluster_one(
                dataset_name,
                model_name,
                max_documents,
                check_qucosa_download
            )
        else:
            raise ValueError(f"datasource '{datasource}' not supported for clustering")

    return action


def _experiments_cluster_one_subparser(parser: argparse.ArgumentParser, datasource: str):
    parser.set_defaults(func=_experiments_cluster_one_action_generator(datasource))
    add_logging_arguments(parser)
    add_common_qucosa_arguments(parser)
    add_storage_directory_arguments(parser)

    dataset_names = available_dataset_names_by_datasource(datasource)
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
