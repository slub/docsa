"""Experiments based on artifical data that was randomly generated."""

import logging
import os

from slub_docsa.common.paths import FIGURES_DIR
from slub_docsa.evaluation.pipeline import evaluate_dataset
from slub_docsa.evaluation.data import generate_random_dbpedia_dataset
from slub_docsa.evaluation.plotting import score_matrix_box_plot
from slub_docsa.experiments.default import default_named_models, default_named_scores

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # setup dataset
    dataset = generate_random_dbpedia_dataset("english", 1000, 5)

    # setup models and scores
    model_names, model_classes = default_named_models()
    score_names, score_functions = default_named_scores()

    # do evaluate
    score_matrix = evaluate_dataset(
        n_splits=10,
        dataset=dataset,
        models=model_classes,
        score_functions=score_functions,
        random_state=0
    )

    # generate figure
    score_matrix_box_plot(
        score_matrix,
        model_names,
        score_names,
        columns=2
    ).write_html(
        os.path.join(FIGURES_DIR, "artificial_box_plot.html"),
        include_plotlyjs="cdn",
        # default_height=f"{len(score_names) * 500}px"
    )
