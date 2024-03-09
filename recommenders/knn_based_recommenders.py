"""
knn_based_recommenders.py

This module defines recommendation engines apply some kind of kNN algorithm to estimate ratings.

"""

import logging
from typing import Callable, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from surprise import AlgoBase, PredictionImpossible, Trainset

from dataset.binary_dataset import load_binary_dataset_from_trainset
from dataset.common import convert_trainset_to_matrix
from pattern_mining.common import (
    apply_bicluster_sparsity_filter,
    apply_bicluster_coverage_filter,
    apply_bicluster_relative_size_filter,
)
from pattern_mining.formal_concept_analysis import (
    get_factor_matrices_from_concepts,
    Concept,
    create_concept,
)

from . import DEFAULT_LOGGER
from .common import (
    get_cosine_similarity_matrix,
    get_top_k_biclusters_for_user,
    compute_neighborhood_cosine_similarity,
    get_indices_above_threshold,
)


def merge_biclusters(
    biclusters: List[Concept],
) -> Concept:
    """
    Merges a list of biclusters into a single bicluster. This means that the extent of the new
    bicluster will be the union of the extents of the given biclusters and the intent of the new
    bicluster will be the union of the intents of the given biclusters.

    Args:
        biclusters (List[Concept]): A list of biclusters.

    Returns:
        Concept: A new bicluster that is the result of merging the given biclusters.
    """

    assert len(biclusters) > 0
    assert all(isinstance(bicluster, Concept) for bicluster in biclusters)

    new_bicluster_extent = np.array([], dtype=np.int64)
    new_bicluster_intent = np.array([], dtype=np.int64)

    for bicluster in biclusters:
        new_bicluster_extent = np.union1d(new_bicluster_extent, bicluster.extent)
        new_bicluster_intent = np.union1d(new_bicluster_intent, bicluster.intent)

    return create_concept(new_bicluster_extent, new_bicluster_intent)


def calculate_weighted_rating(
    target_mean: float,
    neighbors_ratings: np.ndarray,
    neighbors_similarities: np.ndarray,
    neighbors_means: np.ndarray,
) -> float:
    """
    Calculates the weighted rating of a target item based on the ratings of its neighbors.

    Args:
        target_mean (float): The mean rating of the target item.
        neighbors_ratings (np.ndarray): An array containing the ratings of the neighbors.
        neighbors_similarities (np.ndarray): An array containing the similarities between the
            target item and its neighbors.
        neighbors_means (np.ndarray): An array containing the mean ratings of the neighbors.

    Note:
        All arrays must be ordered in the same way. That is, the rating, similarity and mean of
        the same neighbor must be at the same position in their respective arrays.

    Returns:
        float: The weighted rating of the target item.

    Raises:
        AssertionError: If any of the following conditions is not met:
            - target_mean is a float.
            - neighbors_ratings is a numpy array of floats.
            - neighbors_similarities is a numpy array of floats.
            - neighbors_means is a numpy array of floats.
            - neighbors_ratings, neighbors_similarities and neighbors_means have the same size.
            - neighbors_ratings, neighbors_similarities and neighbors_means have size greater than
                zero.
            - All similarities are greater than zero and less than or equal to one.

    """
    assert isinstance(target_mean, float)

    assert isinstance(neighbors_ratings, np.ndarray)
    assert neighbors_ratings.dtype == np.float64

    assert isinstance(neighbors_similarities, np.ndarray)
    assert neighbors_similarities.dtype == np.float64
    assert all(0 < similarity <= 1 for similarity in neighbors_similarities)

    assert isinstance(neighbors_means, np.ndarray)
    assert neighbors_means.dtype == np.float64

    assert neighbors_ratings.size == neighbors_similarities.size == neighbors_means.size
    assert neighbors_ratings.size > 0

    prediction = target_mean
    sum_similarities: float = 0.0
    sum_ratings: float = 0.0

    for rating, similarity, item_mean in zip(
        neighbors_ratings, neighbors_similarities, neighbors_means
    ):
        sum_similarities += similarity
        sum_ratings += similarity * (rating - item_mean)

    prediction += sum_ratings / sum_similarities

    return prediction


def get_k_top_neighbors(
    x: int,
    y: int,
    dataset: np.ndarray,
    users_neighborhood: np.ndarray,
    similarity_matrix: np.ndarray,
    means: np.ndarray,
    knn_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the k nearest neighbors on precomputed user's neighborhood.

    Since this operation is symmetric, x and y can represent either users or items and the dataset
    can be either the original dataset or its transpose. Therefore, the user's neighborhood can be
    either the extent or the intent of a bicluster.

    If x represents a user, then the user's neighborhood is the extent of a bicluster and the
    similarity matrix is the similarity matrix between users and we are looking for the k nearest
    users to the user x.

    If x represents an item, then the user's neighborhood is the intent of a bicluster and the
    similarity matrix is the similarity matrix between items and we are looking for the k nearest
    items to the item x.

    Args:
        x (int): The index of the target user or item.
        y (int): The index of the target item or user.
        dataset (np.ndarray): The original dataset or its transpose.
        users_neighborhood (np.ndarray): The extent or intent of a bicluster.
        similarity_matrix (np.ndarray): The precomputed similarity matrix between users or items.
        means (np.ndarray): The mean ratings of users or items.
        knn_k (int): The number of nearest neighbors to return.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the ratings, similarities
            and means of the k nearest neighbors. Neighbors are ordered by similarity in descending
            order. If there are no neighbors, then the arrays will be empty.

    Raises:
        AssertionError: If any of the following conditions is not met:
            - x is not an int.
            - x is negative.
            - y is not an int.
            - y is negative.
            - dataset is not a numpy array.
            - dataset has the wrong dtype.
            - dataset has the wrong number of dimensions.
            - users_neighborhood is not a numpy array.
            - users_neighborhood has the wrong dtype.
            - users_neighborhood has the wrong number of dimensions.
            - users_neighborhood is empty.
            - similarity_matrix is not a numpy array.
            - similarity_matrix has the wrong dtype.
            - similarity_matrix has the wrong number of dimensions.
            - similarity_matrix is not square.
            - similarity_matrix has the wrong number of rows.
            - means is not a numpy array.
            - means has the wrong dtype.
            - means has the wrong number of dimensions.
            - means has the wrong number of rows.
            - knn_k is not an int.
            - knn_k is not positive.
    """

    def validate_inputs(
        x: int,
        y: int,
        dataset: np.ndarray,
        users_neighborhood: np.ndarray,
        similarity_matrix: np.ndarray,
        means: np.ndarray,
        knn_k: int,
    ):
        assert isinstance(x, int), f"x is not an int: {x}"
        assert x >= 0, f"x is negative: {x}"

        assert isinstance(y, int), f"y is not an int: {y}"
        assert y >= 0, f"y is negative: {y}"

        assert isinstance(dataset, np.ndarray)
        assert dataset.dtype == np.float64, f"dataset has wrong dtype: {dataset.dtype}"
        assert dataset.ndim == 2, f"dataset has wrong number of dimensions: {dataset.ndim}"

        assert isinstance(users_neighborhood, np.ndarray)
        assert (
            users_neighborhood.dtype == np.int64
        ), f"user_neighborhood has wrong dtype: {users_neighborhood.dtype}"
        assert (
            users_neighborhood.ndim == 1
        ), f"user_neighborhood has wrong number of dimensions: {users_neighborhood.ndim}"
        assert users_neighborhood.size > 0, f"user_neighborhood is empty: {users_neighborhood.size}"

        assert isinstance(similarity_matrix, np.ndarray)
        assert (
            similarity_matrix.dtype == np.float64
        ), f"similarity_matrix has wrong dtype: {similarity_matrix.dtype}"
        assert (
            similarity_matrix.ndim == 2
        ), f"similarity_matrix has wrong number of dimensions: {similarity_matrix.ndim}"
        assert (
            similarity_matrix.shape[0] == similarity_matrix.shape[1]
        ), f"similarity_matrix is not square: {similarity_matrix.shape}"
        assert (
            similarity_matrix.shape[0] == dataset.shape[0]
        ), f"similarity_matrix has wrong number of rows: {similarity_matrix.shape[0]}"

        assert isinstance(means, np.ndarray), f"means is not a numpy array: {means}"
        assert means.dtype == np.float64, f"means has wrong dtype: {means.dtype}"
        assert means.ndim == 1, f"means has wrong number of dimensions: {means.ndim}"
        assert means.size == dataset.shape[0], f"means has wrong number of rows: {means.size}"

        assert isinstance(knn_k, int), f"knn_k is not an int: {knn_k}"
        assert knn_k > 0, f"knn_k is not positive: {knn_k}"

        assert x not in users_neighborhood, f"x can't be on the neighborhood: {x}"

    validate_inputs(x, y, dataset, users_neighborhood, similarity_matrix, means, knn_k)

    neighborhood_similarity = similarity_matrix[x, users_neighborhood]
    neighborhood_ratings = dataset[users_neighborhood, y]
    neighborhood_means = means[users_neighborhood]

    valid_mask = (
        (~np.isnan(neighborhood_ratings))
        & (~np.isnan(neighborhood_similarity))
        & (neighborhood_similarity != 0)
    )

    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])

    valid_neighborhood_similarity = neighborhood_similarity[valid_mask]
    valid_neighborhood_ratings = neighborhood_ratings[valid_mask]
    valid_neighborhood_means = neighborhood_means[valid_mask]

    k_top_neighbors_indices_asc = np.argsort(valid_neighborhood_similarity)[
        -min(knn_k, len(valid_neighborhood_similarity)) :
    ]
    k_top_neighbors_indices_desc = k_top_neighbors_indices_asc[::-1]

    k_top_neighbors_ratings = valid_neighborhood_ratings[k_top_neighbors_indices_desc]
    k_top_neighbors_similarity = valid_neighborhood_similarity[k_top_neighbors_indices_desc]
    k_top_means = valid_neighborhood_means[k_top_neighbors_indices_desc]

    return k_top_neighbors_ratings, k_top_neighbors_similarity, k_top_means


# pylint: disable=C0103
class KNNOverLatentSpaceRecommender(AlgoBase, ABC):
    """
    KNNOverLatentSpaceRecommender is an abstract class for recommendation engines
    based on the KNN algorithm. However, instead of using the original dataset,
    these recommendation engines use a latent dataset generated from the original
    dataset using formal concept analysis (FCA). It extends the functionality of
    the AlgoBase class from the Surprise library and provides methods for
    generating recommendations.
    """

    def __init__(
        self,
        dataset_binarization_threshold: float = 1.0,
        knn_k: int = 30,
        knn_similarity_matrix_strategy: Callable = get_cosine_similarity_matrix,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        self.logger = logger

        # Dataset binarization attributes
        self.dataset_binarization_threshold = dataset_binarization_threshold
        self.binary_dataset = None

        # Pattern extraction attributes
        self.formal_context: List[Concept] = []
        self.number_of_factors = None
        self.A = None
        self.B = None
        self.sim = None  # Similarity matrix

        # KNN attributes
        self.knn_k = knn_k
        self.knn_distance_strategy = knn_similarity_matrix_strategy

    @abstractmethod
    def generate_formal_context(self):
        """
        Generates the formal context from the patterns extracted from the training data.

        Override this method in a subclass to implement the desired pattern extraction.
        This method should set the formal_context attribute based on the patterns extracted
        from the training data at self.binary_dataset.
        """

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """
        AlgoBase.fit(self, trainset)

        # Generate binary dataset
        self.logger.debug("Generating binary dataset...")

        self.binary_dataset = load_binary_dataset_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )
        self.logger.debug("Generating binary dataset OK")

        # Generate formal context
        self.logger.debug("Generating formal context...")
        self.generate_formal_context()
        self.number_of_factors = len(self.formal_context)

        if self.number_of_factors == 0:
            raise ValueError("No factors were extracted from the dataset.")

        self.logger.debug("Generating formal context OK")

        # Generate similarity matrix
        self.logger.info("Generating Similarity Matrix...")
        self.A, self.B = get_factor_matrices_from_concepts(
            self.formal_context,
            self.binary_dataset.shape[0],
            self.binary_dataset.shape[1],
        )

        self.sim = get_cosine_similarity_matrix(self.A)
        self.logger.info("Generating Similarity Matrix OK")

        return self

    def estimate(self, user: int, item: int):
        """
        Estimates the rating of a given user for a given item. This function is not supposed to be
        called directly since it uses the Surprise's internal user and item ids. Surprise uses this
        callback internally to make predictions. Use the predict() or test() methods instead which
        use the raw user and item ids.

        Args:
            user (int): internal user id.
            item (int): internal item id.

        Returns:
            tuple: A tuple containing the predicted rating and a dictionary with
                additional details.

        Raises:
            PredictionImpossible: If the user and/or item is unknown or if there
                are not enough neighbors to make a prediction.
        """

        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        ruid = self.trainset.to_raw_uid(user)
        riid = self.trainset.to_raw_iid(item)

        self.logger.debug(f"Estimating rating for user {user} ({ruid}) and item {item} ({riid})...")

        # Get a list of tuples (neighbor, similarity, rating) representing all neighbors
        neighbors = [
            (other_user, self.sim[user, other_user], rating)
            for (other_user, rating) in self.trainset.ir[item]
        ]

        # Remove neighbors with no similarity. Absence of similarity is represented by NaN and
        # happens when the similarity between these users was impossible.
        neighbors = [neighbor for neighbor in neighbors if not np.isnan(neighbor[1])]

        # Sort neighbors by similarity in descending order.
        nearest_neighbors = sorted(neighbors, key=lambda d: d[1], reverse=True)

        self.logger.debug(f"Available neighbors: {len(nearest_neighbors)}")

        # Compute the weighted average of the ratings of the k nearest neighbors
        ratings_sum = 0
        weights_sum = 0
        neighbors_used = []

        for neighbor in nearest_neighbors:
            # Stop if we have enough neighbors
            if len(neighbors_used) >= self.knn_k:
                break

            neighbor_iid, neighbor_similarity, neighbor_rating = neighbor
            neighbor_ruid = self.trainset.to_raw_uid(neighbor_iid)

            if neighbor_similarity == 0:
                continue

            ratings_sum += neighbor_similarity * neighbor_rating
            weights_sum += neighbor_similarity
            neighbors_used.append((neighbor_ruid, neighbor_similarity, neighbor_rating))

        if not neighbors_used:
            raise PredictionImpossible("Not enough neighbors.")

        rating = ratings_sum / weights_sum

        self.logger.debug(f"Neighbors used: {len(neighbors_used)}")

        # Additional details
        details = {"actual_k": len(neighbors_used), "neighbors_used": neighbors_used}

        return rating, details


class BiAKNN(AlgoBase, ABC):
    """
    Bicluster aware kNN (BiAKNN) is an abstract class for recommendation engines based on the kNN
    algorithm. However, instead of using the original dataset, these recommendation engines use
    restricts the neighborhood of each item to a union of biclusters. It extends the functionality
    of the AlgoBase class from the Surprise library and provides methods for generating
    recommendations.

    The method compute_biclusters_from_trainset() must be implemented by subclasses. This method
    is responsible for computing the biclusters from the dataset.
    """

    def __init__(
        self,
        minimum_bicluster_sparsity: Optional[float] = None,
        minimum_bicluster_coverage: Optional[float] = None,
        minimum_bicluster_relative_size: Optional[int] = None,
        knn_type: str = "item",
        user_binarization_threshold: float = 1.0,
        number_of_top_k_biclusters: Optional[int] = None,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        assert isinstance(minimum_bicluster_sparsity, float) or minimum_bicluster_sparsity is None
        if minimum_bicluster_sparsity is not None:
            assert 0 <= minimum_bicluster_sparsity <= 1

        assert isinstance(minimum_bicluster_coverage, float) or minimum_bicluster_coverage is None
        if minimum_bicluster_coverage is not None:
            assert 0 <= minimum_bicluster_coverage <= 1

        assert (
            isinstance(minimum_bicluster_relative_size, float)
            or minimum_bicluster_relative_size is None
        )
        if minimum_bicluster_relative_size is not None:
            assert 0 <= minimum_bicluster_relative_size <= 1

        assert knn_type in ["user", "item"]

        assert isinstance(user_binarization_threshold, float)
        assert user_binarization_threshold >= 0

        assert isinstance(number_of_top_k_biclusters, int) or number_of_top_k_biclusters is None
        if number_of_top_k_biclusters is not None:
            assert number_of_top_k_biclusters > 0

        assert isinstance(knn_k, int)
        assert knn_k > 0

        # Bicluster filtering parameters
        self.minimum_bicluster_sparsity = minimum_bicluster_sparsity
        self.minimum_bicluster_coverage = minimum_bicluster_coverage
        self.minimum_bicluster_relative_size = minimum_bicluster_relative_size

        # User-item neighborhood parameters
        self.user_binarization_threshold = user_binarization_threshold
        self.number_of_top_k_biclusters = number_of_top_k_biclusters

        # KNN parameters
        self.knn_type = knn_type
        self.knn_k = knn_k

        # Other internal attributes
        self.logger = logger
        self.dataset = None
        self.neighborhood = {}
        self.similarity_matrix = None
        self.means = None
        self.biclusters = None
        self.n = None

        self.trainset = None

    @abstractmethod
    def compute_biclusters_from_trainset(self) -> List[np.ndarray]:
        """
        This method is responsible for computing the biclusters that will be used to generate the
        user-item neighborhood for each user in the dataset. Subclasses must implement this method.

        This method will be called after the loading of the trainset. Therefore, the trainset
        attribute will be available and should be used to generate the user-item neighborhood.

        The biclusters must be returned as a list of Concepts.

        """

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """

        AlgoBase.fit(self, trainset)

        self.dataset = convert_trainset_to_matrix(trainset)
        self.compute_biclusters_from_trainset()

        self._apply_filters()

        if not self.number_of_top_k_biclusters:
            self.number_of_top_k_biclusters = len(self.biclusters)

        self._generate_neighborhood()
        self._calculate_means()
        self._instantiate_similarity_matrix()

    def _apply_filters(self) -> None:
        """
        Apply filters to the biclusters based on the specified criteria.

        This method applies filters to the biclusters based on the following criteria:
        - minimum bicluster sparsity
        - minimum bicluster coverage
        - minimum bicluster relative size

        If the number of top k biclusters is not specified, it is set to the total number of
        biclusters.
        """
        if self.minimum_bicluster_sparsity:
            self.biclusters = apply_bicluster_sparsity_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_sparsity
            )

        if self.minimum_bicluster_coverage:
            self.biclusters = apply_bicluster_coverage_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_coverage
            )

        if self.minimum_bicluster_relative_size:
            self.biclusters = apply_bicluster_relative_size_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_relative_size
            )

    def _generate_neighborhood(self) -> None:
        """
        Generates the neighborhood for each user based on the dataset and biclusters.

        The neighborhood is determined by selecting the top-k biclusters that are most relevant to
        the user, merging them into a single bicluster, and extracting either the extent or intent
        depending on the knn_type.
        """
        for user_id in range(self.dataset.shape[0]):
            user_as_tidset = get_indices_above_threshold(
                self.dataset[user_id], self.user_binarization_threshold
            )

            merged_bicluster = create_concept([], [])
            if self.number_of_top_k_biclusters:
                top_k_biclusters = get_top_k_biclusters_for_user(
                    self.biclusters, user_as_tidset, self.number_of_top_k_biclusters
                )
                if top_k_biclusters:
                    merged_bicluster = merge_biclusters(top_k_biclusters)

            if self.knn_type == "user":
                neighborhood = merged_bicluster.extent
            else:
                neighborhood = merged_bicluster.intent

            self.neighborhood[user_id] = neighborhood

    def _calculate_means(self):
        """
        Calculate the mean ratings for each user or item.

        If knn_type is "user", calculate the mean ratings for each user.
        If knn_type is "item", calculate the mean ratings for each item.
        """
        if self.knn_type == "user":
            self.n = self.trainset.n_users
            ratings_map = self.trainset.ur.items()
        else:
            self.n = self.trainset.n_items
            ratings_map = self.trainset.ir.items()

        self.means = np.full((self.n), dtype=np.float64, fill_value=np.NAN)
        for ratings_id, ratings in ratings_map:
            self.means[ratings_id] = np.mean([r for (_, r) in ratings])

    def _instantiate_similarity_matrix(self):
        """
        Instantiate the similarity matrix with NaN values.

        This method initializes the similarity matrix with NaN values. The similarity matrix is a
        square matrix of size n x n, where n is the number of items in the dataset. Each element of
        the matrix represents the similarity between two items.
        """
        self.similarity_matrix = np.full((self.n, self.n), dtype=np.float64, fill_value=np.NAN)

    def estimate(self, user: int, item: int):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        if self.knn_type == "user":
            main_index, secondary_index = user, item
            dataset = self.dataset
        else:
            main_index, secondary_index = item, user
            dataset = self.dataset.T

        user_neighborhood = self.neighborhood[user]
        user_neighborhood = np.setdiff1d(user_neighborhood, main_index)

        if user_neighborhood.size == 0:
            raise PredictionImpossible("Not enough neighbors.")

        compute_neighborhood_cosine_similarity(
            dataset, self.similarity_matrix, main_index, user_neighborhood
        )

        k_top_neighbors_ratings, k_top_neighbors_similarity, k_top_means = get_k_top_neighbors(
            main_index,
            secondary_index,
            dataset,
            user_neighborhood,
            self.similarity_matrix,
            self.means,
            self.knn_k,
        )

        if k_top_neighbors_ratings.size == 0:
            raise PredictionImpossible("Not enough neighbors.")

        prediction = calculate_weighted_rating(
            self.means[main_index],
            k_top_neighbors_ratings,
            k_top_neighbors_similarity,
            k_top_means,
        )

        return prediction, {"actual_k": k_top_neighbors_ratings.size}
