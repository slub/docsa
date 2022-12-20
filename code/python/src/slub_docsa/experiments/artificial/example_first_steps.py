"""Run the example code provided as part of the main documentation."""

# pylint: disable=ungrouped-imports


if __name__ == "__main__":

    from slub_docsa.common.document import Document

    documents = [
        Document(uri="uri://document1", title="This is a document title"),
        Document(uri="uri://document2", title="Document with interesting topic"),
        Document(uri="uri://document3", title="A boring topic"),
    ]

    # ------------

    subjects = [
        ["uri://subject1", "uri://subject2"],    # document 1
        ["uri://subject3"],                      # document 2
        ["uri://subject3", "uri://subject4"],    # document 3
    ]

    # ------------

    from slub_docsa.common.dataset import SimpleDataset

    dataset = SimpleDataset(documents=documents, subjects=subjects)

    # ------------

    from slub_docsa.data.preprocess.vectorizer import ScikitTfidfVectorizer

    vectorizer = ScikitTfidfVectorizer()

    # ------------

    from slub_docsa.models.classification.scikit import ScikitClassifier
    from sklearn.neighbors import KNeighborsClassifier

    model = ScikitClassifier(
        predictor=KNeighborsClassifier(n_neighbors=2),
        vectorizer=vectorizer
    )

    # ------------

    from slub_docsa.evaluation.classification.incidence import subject_incidence_matrix_from_targets
    from slub_docsa.evaluation.classification.incidence import unique_subject_order

    subject_order = unique_subject_order(dataset.subjects)
    incidence_matrix = subject_incidence_matrix_from_targets(
        dataset.subjects,
        subject_order
    )

    print(subject_order)
    print(incidence_matrix)

    # ------------

    model.fit(dataset.documents, incidence_matrix)

    # ------------

    new_documents = [
        Document(uri="uri://new_document1", title="Title of the new document"),
        Document(uri="uri://new_document2", title="Another boring topic"),
    ]

    predicted_probabilities = model.predict_proba(new_documents)
    print(predicted_probabilities)

    # ------------

    from slub_docsa.evaluation.classification.incidence import ThresholdIncidenceDecision

    incidence_decision_function = ThresholdIncidenceDecision(threshold=0.5)
    predicted_incidence = incidence_decision_function(predicted_probabilities)
    print(predicted_incidence)

    # ------------

    from slub_docsa.evaluation.classification.incidence import subject_targets_from_incidence_matrix

    predicted_subjects = subject_targets_from_incidence_matrix(predicted_incidence, subject_order)
    print(predicted_subjects)

    # ------------

    from slub_docsa.evaluation.classification.score.scikit import scikit_incidence_metric
    from sklearn.metrics import f1_score

    true_subjects = [
        ["uri://subject1", "uri://subject2"],
        ["uri://subject4"]
    ]

    true_incidence = subject_incidence_matrix_from_targets(true_subjects, subject_order)

    score = scikit_incidence_metric(
        incidence_decision_function, f1_score, average="micro"
    )(
        true_incidence, predicted_probabilities
    )
    print("f1 score is", score)
