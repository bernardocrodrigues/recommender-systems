""" binaps_recommender.py

This module contains all recommenders based on the BinaPS algorithm.
"""

import logging
from typing import Optional
from tempfile import TemporaryDirectory

from pattern_mining.binaps.binaps_wrapper import run_binaps, get_patterns_from_weights
from pattern_mining.formal_concept_analysis import create_concept
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    save_as_binaps_compatible_input,
)

from .knn_based_recommenders import BiAKNN
from . import DEFAULT_LOGGER


class BinaPsKNNRecommender(BiAKNN):
    """
    Recommender class that uses the BinaPs algorithm to generate the patterns that are used
    to generate a user-item neighborhood that is, then, used for generating recommendations.
    """

    def __init__(
        self,
        epochs: int = 100,
        hidden_dimension_neurons_number: Optional[int] = None,
        weights_binarization_threshold: float = 0.2,
        dataset_binarization_threshold: float = 1.0,
        minimum_bicluster_sparsity: Optional[float] = None,
        minimum_bicluster_coverage: Optional[float] = None,
        minimum_bicluster_relative_size: Optional[int] = None,
        knn_type: str = "item",
        user_binarization_threshold: float = 1.0,
        number_of_top_k_biclusters: Optional[int] = None,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        assert isinstance(epochs, int) and epochs > 0
        assert hidden_dimension_neurons_number is None or (
            isinstance(hidden_dimension_neurons_number, int) and hidden_dimension_neurons_number > 0
        )
        assert (
            isinstance(weights_binarization_threshold, float)
            and 0 < weights_binarization_threshold <= 1
        )
        assert (
            isinstance(dataset_binarization_threshold, float)
            and 0 < dataset_binarization_threshold
        )

        super().__init__(
            minimum_bicluster_sparsity=minimum_bicluster_sparsity,
            minimum_bicluster_coverage=minimum_bicluster_coverage,
            minimum_bicluster_relative_size=minimum_bicluster_relative_size,
            knn_type=knn_type,
            user_binarization_threshold=user_binarization_threshold,
            number_of_top_k_biclusters=number_of_top_k_biclusters,
            knn_k=knn_k,
            logger=logger,
        )

        self.epochs = epochs
        self.hidden_dimension_neurons_number = hidden_dimension_neurons_number
        self.weights_binarization_threshold = weights_binarization_threshold
        self.dataset_binarization_threshold = dataset_binarization_threshold

    def compute_biclusters_from_trainset(self):
        binary_dataset = load_binary_dataset_from_trainset(
            self.trainset, threshold=self.dataset_binarization_threshold
        )


        with TemporaryDirectory() as temporary_directory:
            with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
                save_as_binaps_compatible_input(binary_dataset, file_object)

                weights, _, _ = run_binaps(
                    input_dataset_path=file_object.name,
                    epochs=self.epochs,
                    hidden_dimension=self.hidden_dimension_neurons_number,
                )

        patterns = get_patterns_from_weights(weights, self.weights_binarization_threshold)

        self.biclusters = []
        for pattern in patterns:
            concept = create_concept([], pattern)
            self.biclusters.append(concept)
