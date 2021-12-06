"""Experiments command."""

import argparse
from slub_docsa.cli.common import add_logging_arguments, setup_logging_from_args
from slub_docsa.cli.qucosa import available_qucosa_dataset_names, available_qucosa_model_names
from slub_docsa.experiments.qucosa.classify_many import qucosa_experiments_classify_many


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


def _experiments_qucosa_classify_many_action(args):
    """Perform `experiments qucosa classify_many` action."""
    setup_logging_from_args(args)

    dataset_subset = args.datasets
    model_subset = args.models
    n_splits = int(args.cross_splits)

    avaiable_datasets = available_qucosa_dataset_names()
    avaiable_models = available_qucosa_model_names()

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
        load_cached_predictions=False,
        split_function_name="random",
        stop_after_evaluating_split=None
    )


def _experiments_qucosa_classify_many_subparser(parser: argparse.ArgumentParser):
    """Return sub-parser for `experiments qucosa classify_many` command."""
    parser.set_defaults(func=_experiments_qucosa_classify_many_action)
    add_logging_arguments(parser)

    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="a list of dataset variants to evaluate: " + ", ".join(available_qucosa_dataset_names()),
        default=["qucosa_de_titles_langid_rvk", "qucosa_de_abstracts_langid_rvk", "qucosa_de_fulltexts_langid_rvk"],
    )

    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="a list of models to evaluate: " + ", ".join(available_qucosa_model_names(False)),
        default=["nihilistic", "oracle", "tfidf_10k_knn_k=1"],
    )

    parser.add_argument(
        "--cross_splits",
        "-c",
        help="the number of cross-validation splits, default 10",
        default=10,
    )
