"""Run the example code provided as part of the main documentation."""

# pylint: disable=ungrouped-imports, invalid-name


if __name__ == "__main__":

    # ------------

    from slub_docsa.data.artificial.tokens import token_probabilities_from_dbpedia

    token_probabilities = token_probabilities_from_dbpedia("en", n_docs=1000)

    print(len(token_probabilities))

    for t in list(token_probabilities.keys())[:5]:
        print(t, ":", token_probabilities[t])

    # ------------

    from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset_from_token_probabilities

    dataset, subject_hierarchy = generate_hierarchical_random_dataset_from_token_probabilities(
        token_probabilities, n_documents=1000, n_subjects=10
    )

    print(dataset.documents[0])

    # ------------

    from slub_docsa.common.subject import print_subject_hierarchy

    print_subject_hierarchy("en", subject_hierarchy)

    # ------------

    from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
    from slub_docsa.data.preprocess.subject import prune_subject_targets_to_minimum_samples

    min_samples = 10

    dataset.subjects = prune_subject_targets_to_minimum_samples(min_samples, dataset.subjects, subject_hierarchy)
    dataset = filter_subjects_with_insufficient_samples(dataset, min_samples)

    # ------------

    from slub_docsa.evaluation.classification.incidence import unique_subject_order

    subject_order = unique_subject_order(dataset.subjects)
    print(len(subject_order))

    # ------------

    from slub_docsa.models.classification.dummy import NihilisticModel, OracleModel
    from slub_docsa.models.classification.scikit import ScikitClassifier
    from slub_docsa.data.preprocess.vectorizer import StemmingVectorizer, GensimTfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    models = [
        OracleModel(),
        NihilisticModel(),
        ScikitClassifier(
            predictor=KNeighborsClassifier(n_neighbors=1),
            vectorizer=StemmingVectorizer(GensimTfidfVectorizer(max_features=2000), "en"),
        )
    ]

    # ------------

    from slub_docsa.evaluation.classification.score.scikit import scikit_metric_for_best_threshold_based_on_f1score
    from slub_docsa.evaluation.classification.score.scikit import scikit_incidence_metric
    from slub_docsa.evaluation.classification.score.hierarchical import cesa_bianchi_loss_generator
    from slub_docsa.evaluation.classification.incidence import PositiveTopkIncidenceDecision

    scores = [
        # f1 score for best threshold
        scikit_metric_for_best_threshold_based_on_f1score(
            f1_score, average="micro", zero_division=0
        ),
        # f1 score for top-3 selection
        scikit_incidence_metric(
            PositiveTopkIncidenceDecision(3),
            f1_score,
            average="micro",
            zero_division=0
        ),
        # hierarchical loss
        scikit_metric_for_best_threshold_based_on_f1score(
            cesa_bianchi_loss_generator(subject_hierarchy, subject_order, log_factor=1000),
        )
    ]

    # ------------

    import logging
    logging.basicConfig(level=logging.INFO)

    # ------------

    from slub_docsa.evaluation.classification.pipeline import score_classification_models_for_dataset
    from slub_docsa.evaluation.classification.split import scikit_kfold_splitter

    n_splits = 10
    split_function = scikit_kfold_splitter(n_splits)

    score_matrix, _ = score_classification_models_for_dataset(
        n_splits,
        dataset,
        subject_order,
        models,
        split_function,
        scores,
        [],
    )

    # ------------

    from slub_docsa.evaluation.classification.plotting import score_matrix_box_plot
    from slub_docsa.evaluation.classification.plotting import write_multiple_figure_formats

    figure = score_matrix_box_plot(
        score_matrix,
        model_names=["oracle", "nihilistic", "knn k=1"],
        score_names=["t=best f1_score", "top-3 f1_score", "h-loss"],
        score_ranges=[(0, 1), (0, 1), (0, None)]
    )

    write_multiple_figure_formats(figure, filepath="example_score_matrix")
