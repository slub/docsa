"""Test various training / test split methods."""

from slub_docsa.common.dataset import Dataset
from slub_docsa.data.artificial.simple import generate_random_dataset

from slub_docsa.evaluation.condition import check_dataset_subject_distribution
from slub_docsa.evaluation.condition import check_subject_targets_have_minimum_samples
from slub_docsa.evaluation.split import scikit_kfold_splitter, skmultilearn_iterative_stratification_splitter


def _generate_random_dataset(n_folds, n_tokens, n_docs, n_subjects) -> Dataset:
    dataset = None
    dataset_valid = False
    i = 0
    while not dataset_valid and i < 10:
        dataset = generate_random_dataset(n_tokens, n_docs, n_subjects)
        dataset_valid = check_subject_targets_have_minimum_samples(dataset.subjects, n_folds)
        i += 1

    if dataset is None:
        raise RuntimeError("should never happen")

    if not dataset_valid:
        raise RuntimeError("could not generate valid random dataset")

    return dataset


def _check_folder_for_unblanace_targets_distribution(n_repeats, n_folds, n_docs, n_subjects, split_function):
    for _ in range(n_repeats):
        # generate random dataset
        dataset = _generate_random_dataset(n_folds, 100, n_docs, n_subjects)

        # check each split for balanced targets distribution
        for train_dataset, test_dataset in split_function(dataset):
            if not check_dataset_subject_distribution(train_dataset, test_dataset, (0.5 / n_folds, 2.0 / n_folds)):
                # stop checking as soon as one unbalanced distribution was found
                return False

    # report that folder is balanced for random data
    return True


def test_scikit_kfold_is_unbalanced_multi_label_splitter():
    """Verify that scikit's KFold method is not a multi-label well-balanced stratified splitting method."""
    n_folds = 10
    split_function = scikit_kfold_splitter(n_splits=n_folds)
    assert not _check_folder_for_unblanace_targets_distribution(20, n_folds, 1000, 10, split_function)


def test_skmultilearn_iterative_stratification_is_balanced_multi_label_splitter():
    """Verify that skmultilearn's implementation of IterativeStratification is a well-balanced splitting method."""
    n_folds = 10
    split_function = skmultilearn_iterative_stratification_splitter(n_splits=n_folds)
    assert _check_folder_for_unblanace_targets_distribution(20, n_folds, 1000, 10, split_function)
