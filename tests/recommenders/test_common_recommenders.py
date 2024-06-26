# """
# Tests for the common implementations of the recommenders module.
# """

import math

from unittest.mock import patch
import numpy as np

# import pandas as pd
# import numba as nb
import scipy
import pytest
from recommenders.common import (
    cosine_similarity,
    adjusted_cosine_similarity,
    user_pattern_similarity,
    weight_frequency,
    get_similarity,
    get_similarity_matrix,
    get_top_k_biclusters_for_user,
    get_indices_above_threshold,
    merge_biclusters,
)

from pattern_mining.formal_concept_analysis import create_concept, concepts_are_equal

# pylint: disable=missing-function-docstring


class TestCosineSimilarity:

    def test_no_nan_vaules(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1], dtype=np.float64)
        v = np.array([1], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([0, 0], dtype=np.float64)
        v = np.array([0, 0], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 0
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 0

    def test_one_vector_has_more_items(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = cosine_similarity.py_func(u, v)
        assert similarity == 1

    def test_vector_only_has_nans(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

        u = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

    @pytest.mark.parametrize("execution_number", range(1000))
    def test_partial_similarity_with_no_nans(self, execution_number):
        size = np.random.default_rng().integers(1, 100)

        u = np.random.default_rng().random(size) * 5
        v = np.random.default_rng().random(size) * 5

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, 1 - scipy.spatial.distance.cosine(u, v), rel_tol=1e-9)
        similarity = cosine_similarity.py_func(u, v)
        assert math.isclose(similarity, 1 - scipy.spatial.distance.cosine(u, v), rel_tol=1e-9)

    def test_partial_similarity_with_nans_1(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([1, 2], dtype=np.float64)
        v_no_nan = np.array([1, 2], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )
        similarity = cosine_similarity.py_func(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

    def test_partial_similarity_with_nans_2(self):

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([1], dtype=np.float64)
        v_no_nan = np.array([1], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )
        similarity = cosine_similarity.py_func(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

    def test_partial_similarity_with_nans_3(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, 2, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([3], dtype=np.float64)
        v_no_nan = np.array([2], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )
        similarity = cosine_similarity.py_func(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

    def test_partial_similarity_with_nans_4(self):

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, 1, 1], dtype=np.float64)

        u_no_nan = np.array([4, 5], dtype=np.float64)
        v_no_nan = np.array([1, 1], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )
        similarity = cosine_similarity.py_func(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

    def test_partial_similarity_against_wolfram_1(self):

        u = np.array([2, 3, 1], dtype=np.float64)
        v = np.array([1, 2, 5], dtype=np.float64)

        wolfram_result = 13 / (2 * math.sqrt(105))

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, wolfram_result, rel_tol=1e-8)

    def test_partial_similarity_against_wolfram_2(self):

        u = np.array([1, 2, 2, 5, 5], dtype=np.float64)
        v = np.array([5, 5, 3, 5, 5], dtype=np.float64)

        wolfram_result = 71 / math.sqrt(6431)

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, wolfram_result, rel_tol=1e-8)

    def test_partial_similarity_against_wolfram_3(self):

        u = np.array([1, 2, 2, 4, 5, 5, 5], dtype=np.float64)
        v = np.array([5, 2, 3, 5, 5, 1, 2], dtype=np.float64)

        wolfram_result = 5 / 2 * math.sqrt(3 / 31)

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, wolfram_result, rel_tol=1e-8)


class TestAdjustedCosineSimilarity:

    def test_similarity_no_nan_vaules_1(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 1

    def test_similarity_no_nan_vaules_2(self):
        u = np.array([1], dtype=np.float64)
        v = np.array([1], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 0
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 0

    def test_similarity_no_nan_vaules_3(self):
        u = np.array([1, 2], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 1

    def test_one_vector_has_more_items(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == 0
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == 0

        u = np.array([4, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 3, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert similarity == -1
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert similarity == -1

        u = np.array([4, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 3, 3, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert math.isclose(similarity, -0.8660254037844387, rel_tol=1e-9)
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert math.isclose(similarity, -0.8660254037844387, rel_tol=1e-9)

    def test_vector_only_has_nans(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

        u = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = adjusted_cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = adjusted_cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

    @pytest.mark.parametrize("execution_number", range(1000))
    def test_partial_similarity_with_no_nans(self, execution_number):
        size = np.random.default_rng().integers(2, 100)

        u = np.random.default_rng().random(size) * 5
        v = np.random.default_rng().random(size) * 5

        u_norm = u - np.mean(u)
        v_norm = v - np.mean(v)

        similarity = adjusted_cosine_similarity(u, v)

        assert math.isclose(similarity, adjusted_cosine_similarity(u_norm, v_norm), rel_tol=1e-9)
        assert math.isclose(similarity, adjusted_cosine_similarity.py_func(u, v), rel_tol=1e-9)
        assert math.isclose(similarity, cosine_similarity(u_norm, v_norm), rel_tol=1e-9)


class TestUserPatternSimilarity:

    def test_no_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [6, 7, 8])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [6, 7, 8])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = create_concept([], [])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([], [])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = create_concept([], [2])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 0

    def test_full_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([], [1, 2, 3, 4, 5])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([1, 2, 3, 4], [1])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_full_similarity_but_user_has_more_items(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3, 4], [1, 2, 3, 4])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2], [1])
        similarity = user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_partial_similarity_where_pattern_has_more_items(self):
        user = np.array([1, 2, 3, 4], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3, 4, 5])
        similarity = user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.8, rel_tol=1e-9)

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3, 4, 5])
        similarity = user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3, 4, 5])
        similarity = user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.2, rel_tol=1e-9)

    def test_partial_similarity_where_user_and_pattern_have_some_items_in_common(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2], [4, 5, 6, 7, 8])
        similarity = user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.4, rel_tol=1e-9)

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [1, 2, 3, 6, 7])
        similarity = user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)


class TestWeightFrequency:

    def test_no_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [6, 7, 8])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [6, 7, 8])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = create_concept([], [])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([], [])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = create_concept([], [2])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

    def test_full_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([], [1, 2, 3, 4, 5])
        similarity = weight_frequency(user, pattern)
        assert similarity == 0

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([1, 2, 3, 4], [1])
        similarity = weight_frequency(user, pattern)
        assert similarity == 4

        user = np.array([1, 2], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2])
        similarity = weight_frequency(user, pattern)
        assert similarity == 2

    def test_full_similarity_but_user_has_more_items(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3, 4], [1, 2, 3, 4])
        similarity = weight_frequency(user, pattern)
        assert similarity == 4

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 5], [1, 2, 3])
        similarity = weight_frequency(user, pattern)
        assert similarity == 3

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2], [1])
        similarity = weight_frequency(user, pattern)
        assert similarity == 2

    def test_partial_similarity_where_pattern_has_more_items(self):
        user = np.array([1, 2, 3, 4], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3, 4, 5])
        similarity = weight_frequency(user, pattern)
        assert math.isclose(similarity, 1.6, rel_tol=1e-9)

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = create_concept([1, 2, 3], [1, 2, 3, 4, 5])
        similarity = weight_frequency(user, pattern)
        assert math.isclose(similarity, 1.8, rel_tol=1e-9)

        user = np.array([1], dtype=np.int64)
        pattern = create_concept([1, 2], [1, 2, 3, 4, 5])
        similarity = weight_frequency(user, pattern)
        assert math.isclose(similarity, 0.4, rel_tol=1e-9)

    def test_partial_similarity_where_user_and_pattern_have_some_items_in_common(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2], [4, 5, 6, 7, 8])
        similarity = weight_frequency(user, pattern)
        assert math.isclose(similarity, 0.8, rel_tol=1e-9)

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = create_concept([1, 2, 3, 4, 5, 6], [1, 2, 3, 6, 7])
        similarity = weight_frequency(user, pattern)
        assert math.isclose(similarity, 3.6, rel_tol=1e-9)


class TestGetSimilarity:

    def test_similarity_already_computed(self):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=np.float64)
        similarity_matrix = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )

        similarity = get_similarity(0, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.1, rel_tol=1e-9)

        similarity = get_similarity(0, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.2, rel_tol=1e-9)

        similarity = get_similarity(0, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.3, rel_tol=1e-9)

        similarity = get_similarity(1, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.4, rel_tol=1e-9)

        similarity = get_similarity(1, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.5, rel_tol=1e-9)

        similarity = get_similarity(1, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)

        similarity = get_similarity(2, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.7, rel_tol=1e-9)

        similarity = get_similarity(2, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.8, rel_tol=1e-9)

        similarity = get_similarity(2, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, 0.9, rel_tol=1e-9)

    def test_similarity_not_computed_and_indices_are_equal(self):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=np.float64)
        similarity_matrix = np.array(
            [[np.nan, 0.2, 0.3], [0.4, np.nan, 0.6], [0.7, 0.8, np.nan]], dtype=np.float64
        )

        similarity = get_similarity(0, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, 1, rel_tol=1e-9)

        similarity = get_similarity(1, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, 1, rel_tol=1e-9)

        similarity = get_similarity(2, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, 1, rel_tol=1e-9)

    def test_similarity_not_computed_and_indices_are_not_equal(self):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64)
        similarity_matrix = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        similarity = get_similarity(0, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, 1, rel_tol=1e-9)
        assert math.isclose(similarity, cosine_similarity(dataset[0], dataset[0]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[0, 0], rel_tol=1e-9)

        similarity = get_similarity(0, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[0], dataset[1]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[0, 1], rel_tol=1e-9)

        similarity = get_similarity(0, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[0], dataset[2]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[0, 2], rel_tol=1e-9)

        similarity = get_similarity(0, 3, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[0], dataset[3]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[0, 3], rel_tol=1e-9)

        similarity = get_similarity(1, 0, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[1], dataset[0]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[1, 0], rel_tol=1e-9)

        similarity = get_similarity(1, 1, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[1], dataset[1]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[1, 1], rel_tol=1e-9)

        similarity = get_similarity(1, 2, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[1], dataset[2]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[1, 2], rel_tol=1e-9)

        similarity = get_similarity(1, 3, dataset, similarity_matrix)
        assert math.isclose(similarity, cosine_similarity(dataset[1], dataset[3]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[1, 3], rel_tol=1e-9)

    @patch("recommenders.common.cosine_similarity")
    def test_similarity_not_computed_and_indices_are_not_equal_with_mocks(
        self, cosine_similarity_mock
    ):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64)
        similarity_matrix = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        similarity = get_similarity.py_func(
            0, 0, dataset, similarity_matrix, cosine_similarity_mock
        )
        cosine_similarity_mock.assert_not_called()
        assert math.isclose(similarity, 1, rel_tol=1e-9)
        assert math.isclose(similarity, cosine_similarity(dataset[0], dataset[0]), rel_tol=1e-9)
        assert math.isclose(similarity, similarity_matrix[0, 0], rel_tol=1e-9)

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 99

        similarity = get_similarity.py_func(
            0, 1, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 1
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][0], dataset[0])
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][1], dataset[1])
        assert similarity == 99
        assert similarity_matrix[0, 1] == 99

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 105

        similarity = get_similarity.py_func(
            0, 2, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 1
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][0], dataset[0])
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][1], dataset[2])
        assert similarity == 105
        assert similarity_matrix[0, 2] == 105

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 222

        similarity = get_similarity.py_func(
            0, 3, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 1
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][0], dataset[0])
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][1], dataset[3])
        assert similarity == 222
        assert similarity_matrix[0, 3] == 222

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 11

        similarity = get_similarity.py_func(
            1, 0, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 0
        assert similarity == 99
        assert similarity_matrix[1, 0] == 99

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 23

        similarity = get_similarity.py_func(
            1, 1, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 0
        assert similarity == 1
        assert similarity_matrix[1, 1] == 1

        cosine_similarity_mock.reset_mock()
        cosine_similarity_mock.return_value = 44

        similarity = get_similarity.py_func(
            1, 2, dataset, similarity_matrix, cosine_similarity_mock
        )
        assert cosine_similarity_mock.call_count == 1
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][0], dataset[1])
        np.testing.assert_array_equal(cosine_similarity_mock.call_args[0][1], dataset[2])
        assert similarity == 44
        assert similarity_matrix[1, 2] == 44


class TestGetSimilarityMatrix:

    def test_computed_from_wolfram(self):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=np.float64)

        similarity_matrix = get_similarity_matrix(dataset)

        expected_similarity_matrix = np.array(
            [
                [1, (29 / 854) * math.sqrt(854), 25 / math.sqrt(679)],
                [(29 / 854) * math.sqrt(854), 1, 107 / math.sqrt(11834)],
                [25 / math.sqrt(679), 107 / math.sqrt(11834), 1],
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(similarity_matrix, expected_similarity_matrix, rtol=1e-9)

    @patch("recommenders.common.get_similarity")
    def test_with_mock(self, get_similarity_mock):

        dataset = np.array([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=np.float64)

        get_similarity_mock.side_effect = [1, 2, 3, 4, 5, 6]

        get_similarity_matrix.py_func(dataset, get_similarity_mock)

        assert len(get_similarity_mock.call_args_list) == 6

        call_1_args = get_similarity_mock.call_args_list[0][0]
        assert call_1_args[0] == 0
        assert call_1_args[1] == 0
        np.testing.assert_array_equal(call_1_args[2], dataset)
        np.testing.assert_array_equal(call_1_args[3], np.full((3, 3), np.nan))
        assert call_1_args[4] == get_similarity_mock

        call_2_args = get_similarity_mock.call_args_list[1][0]
        assert call_2_args[0] == 0
        assert call_2_args[1] == 1
        np.testing.assert_array_equal(call_2_args[2], dataset)
        np.testing.assert_array_equal(call_2_args[3], np.full((3, 3), np.nan))
        assert call_2_args[4] == get_similarity_mock

        call_3_args = get_similarity_mock.call_args_list[2][0]
        assert call_3_args[0] == 0
        assert call_3_args[1] == 2
        np.testing.assert_array_equal(call_3_args[2], dataset)
        np.testing.assert_array_equal(call_3_args[3], np.full((3, 3), np.nan))
        assert call_3_args[4] == get_similarity_mock

        call_4_args = get_similarity_mock.call_args_list[3][0]
        assert call_4_args[0] == 1
        assert call_4_args[1] == 1
        np.testing.assert_array_equal(call_4_args[2], dataset)
        np.testing.assert_array_equal(call_4_args[3], np.full((3, 3), np.nan))
        assert call_4_args[4] == get_similarity_mock

        call_5_args = get_similarity_mock.call_args_list[4][0]
        assert call_5_args[0] == 1
        assert call_5_args[1] == 2
        np.testing.assert_array_equal(call_5_args[2], dataset)
        np.testing.assert_array_equal(call_5_args[3], np.full((3, 3), np.nan))
        assert call_5_args[4] == get_similarity_mock

        call_6_args = get_similarity_mock.call_args_list[5][0]
        assert call_6_args[0] == 2
        assert call_6_args[1] == 2
        np.testing.assert_array_equal(call_6_args[2], dataset)
        np.testing.assert_array_equal(call_6_args[3], np.full((3, 3), np.nan))
        assert call_6_args[4] == get_similarity_mock


class TestGetTopKBiclustersForUser:
    A = create_concept([1, 1], [1, 1])
    B = create_concept([2, 2], [2, 2])
    C = create_concept([3, 3], [3, 3])
    D = create_concept([4, 4], [4, 4])

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_1(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 2, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.B)
        assert concepts_are_equal(top_k_biclusters[1], self.A)

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_2(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 2, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.C)
        assert concepts_are_equal(top_k_biclusters[1], self.D)

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_3(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 3, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 3
        assert concepts_are_equal(top_k_biclusters[0], self.B)
        assert concepts_are_equal(top_k_biclusters[1], self.C)
        assert concepts_are_equal(top_k_biclusters[2], self.D)

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_4(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.2, 0.8, 0.6, 0.4]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 1, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 1
        assert concepts_are_equal(top_k_biclusters[0], self.B)

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_5(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.2, 0.8, 0.6, 0.4]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 2, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.C)
        assert concepts_are_equal(top_k_biclusters[1], self.B)

    @patch("recommenders.common.user_pattern_similarity")
    def test__get_top_k_biclusters_for_user_5(self, user_pattern_similarity_mock):
        user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.8, 0.6]

        top_k_biclusters = get_top_k_biclusters_for_user.py_func(
            [self.A, self.B, self.C, self.D], np.array([0]), 2, user_pattern_similarity_mock
        )

        assert len(user_pattern_similarity_mock.call_args_list) == 4

        call_1_args = user_pattern_similarity_mock.call_args_list[0][0]
        np.testing.assert_array_equal(call_1_args[0], np.array([0]))
        np.testing.assert_array_equal(call_1_args[1], self.A)

        call_2_args = user_pattern_similarity_mock.call_args_list[1][0]
        np.testing.assert_array_equal(call_2_args[0], np.array([0]))
        np.testing.assert_array_equal(call_2_args[1], self.B)

        call_3_args = user_pattern_similarity_mock.call_args_list[2][0]
        np.testing.assert_array_equal(call_3_args[0], np.array([0]))
        np.testing.assert_array_equal(call_3_args[1], self.C)

        call_4_args = user_pattern_similarity_mock.call_args_list[3][0]
        np.testing.assert_array_equal(call_4_args[0], np.array([0]))

        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.D)
        assert concepts_are_equal(top_k_biclusters[1], self.C)


class TestGetIndicesAboveThreshold:

    def test_no_index_is_returned(self):
        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 4.0)
        assert len(indices) == 0

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 3.1)
        assert len(indices) == 0

        indices = get_indices_above_threshold(np.array([1, 2], dtype=np.float64), 3.0)
        assert len(indices) == 0

        indices = get_indices_above_threshold(
            np.array([1, 2, 3, 4, 3, 4, 3, 1, 2, 3], dtype=np.float64), 4.5
        )
        assert len(indices) == 0

    def test_some_indices_are_returned(self):
        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 3.0)
        assert len(indices) == 1
        assert indices[0] == 2

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 2.0)
        assert len(indices) == 2
        assert indices[0] == 1
        assert indices[1] == 2

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 2.5)
        assert len(indices) == 1
        assert indices[0] == 2

        indices = get_indices_above_threshold(
            np.array([1, 2, 3, 4, 3, 4, 3, 1, 2, 3], dtype=np.float64), 3.0
        )
        assert len(indices) == 6
        assert indices[0] == 2
        assert indices[1] == 3
        assert indices[2] == 4
        assert indices[3] == 5
        assert indices[4] == 6
        assert indices[5] == 9

    def test_all_indices_are_returned(self):
        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 0.0)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 2

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 1.0)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 2

        indices = get_indices_above_threshold(
            np.array([1, 2, 3, 4, 3, 4, 3, 1, 2, 3], dtype=np.float64), 1.0
        )
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 2
        assert indices[3] == 3
        assert indices[4] == 4
        assert indices[5] == 5
        assert indices[6] == 6
        assert indices[7] == 7
        assert indices[8] == 8
        assert indices[9] == 9


class TestMergeBiclusters:

    def test_no_overlap_1(self):
        biclusters = [create_concept([1, 2], [1, 2, 3]), create_concept([3, 4], [4, 5, 6])]
        merged_bicluster = merge_biclusters(biclusters)

        assert concepts_are_equal(
            merged_bicluster, create_concept([1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
        )

    def test_no_overlap_2(self):
        biclusters = [
            create_concept([1, 2, 3], [1, 2, 3, 4, 5]),
            create_concept([4, 5, 6], [6, 7, 8, 9]),
        ]
        merged_bicluster = merge_biclusters(biclusters)

        assert concepts_are_equal(
            merged_bicluster, create_concept([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        )

    def test_no_overlap_3(self):
        biclusters = [create_concept([1, 2], [1, 2]), create_concept([4, 5], [4, 5])]
        merged_bicluster = merge_biclusters(biclusters)

        assert concepts_are_equal(merged_bicluster, create_concept([1, 2, 4, 5], [1, 2, 4, 5]))

    def test_overlap_1(self):
        biclusters = [create_concept([1, 2], [1, 2]), create_concept([2, 3], [2, 3])]
        merged_bicluster = merge_biclusters(biclusters)

        assert concepts_are_equal(merged_bicluster, create_concept([1, 2, 3], [1, 2, 3]))

    def test_overlap_2(self):
        biclusters = [
            create_concept([1, 2, 5, 6], [1, 2, 7, 8, 9]),
            create_concept([1, 2], [9, 10]),
        ]
        merged_bicluster = merge_biclusters(biclusters)

        assert concepts_are_equal(
            merged_bicluster, create_concept([1, 2, 5, 6], [1, 2, 7, 8, 9, 10])
        )


# class TestComputeNeighborhoodCosineSimilarity:
#     class TestInvalidArgs:
#         def test_1(self):
#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     "not a numpy array", np.array([1, 2, 3]), 0, np.array([1, 2, 3])
#                 )

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     np.array([1, 2, 3]), "not a numpy array", 0, np.array([1, 2, 3])
#                 )

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     np.array([1, 2, 3]), np.array([1, 2, 3]), "not an integer", np.array([1, 2, 3])
#                 )

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     np.array([1, 2, 3]), np.array([1, 2, 3]), 0, "not a numpy array"
#                 )

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     np.array([1, 2, 3]), np.array([1, 2, 3]), 0, np.array([1, 2, 3])
#                 )

#         def test_2(self):
#             dataset = np.array([])
#             similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#             target = 0
#             neighborhood = np.array([1])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#         def test_3(self):
#             dataset = np.array([[1, 2], [3, 4]])
#             similarity_matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8]])
#             target = 0
#             neighborhood = np.array([1])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#         def test_4(self):
#             dataset = np.array([[1, 2], [3, 4]])
#             similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#             target = 2
#             neighborhood = np.array([1])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#         def test_5(self):
#             dataset = np.array([[1, 2], [3, 4]])
#             similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#             target = 0
#             neighborhood = np.array([0, 1])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#         def test_6(self):
#             dataset = np.array([[1, 2], [3, 4]])
#             similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#             target = 0
#             neighborhood = np.array([])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#         def test_7(self):
#             dataset = np.array([[1, 2], [3, 4]])
#             similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
#             target = 0
#             neighborhood = np.array([2])

#             with pytest.raises(AssertionError):
#                 compute_neighborhood_cosine_similarity(
#                     dataset, similarity_matrix, target, neighborhood
#                 )

#     class TestUserScenario:
#         @patch("recommenders.common.cosine_similarity")
#         def test_1(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 0
#             neighborhood = np.array([1, 2])

#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2
#             assert (calls[0].kwargs["u"] == np.array([1, 2, 3])).all()
#             assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()
#             assert (calls[1].kwargs["u"] == np.array([1, 2, 3])).all()
#             assert (calls[1].kwargs["v"] == np.array([7, 8, 9])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[0][2], 0.2)
#             assert np.isclose(similarity_matrix[2][0], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_2(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 1
#             neighborhood = np.array([0, 2])

#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2
#             assert (calls[0].kwargs["u"] == np.array([4, 5, 6])).all()
#             assert (calls[0].kwargs["v"] == np.array([1, 2, 3])).all()
#             assert (calls[1].kwargs["u"] == np.array([4, 5, 6])).all()
#             assert (calls[1].kwargs["v"] == np.array([7, 8, 9])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[1][2], 0.2)
#             assert np.isclose(similarity_matrix[2][1], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_3(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.23, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 2
#             neighborhood = np.array([1])

#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list

#             assert len(calls) == 1
#             assert (calls[0].kwargs["u"] == np.array([7, 8, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()

#             assert np.isclose(similarity_matrix[1][2], 0.23)
#             assert np.isclose(similarity_matrix[2][1], 0.23)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_4(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,  2,  3,  4],
#                                 [5,  6,  7,  8],
#                                 [9, 10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 0
#             neighborhood = np.array([1, 2])

#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2

#             assert (calls[0].kwargs["u"] == np.array([1, 2, 3, 4])).all()
#             assert (calls[0].kwargs["v"] == np.array([5, 6, 7, 8])).all()

#             assert (calls[1].kwargs["u"] == np.array([1, 2, 3, 4])).all()
#             assert (calls[1].kwargs["v"] == np.array([9, 10, 11, 12])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[0][2], 0.2)
#             assert np.isclose(similarity_matrix[2][0], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_5(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3],
#                                 [4,   5,  6],
#                                 [7,   8,  9],
#                                 [10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((4, 4), np.nan)

#             target = 0

#             neighborhood = np.array([1, 3])
#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2

#             assert (calls[0].kwargs["u"] == np.array([1, 2, 3])).all()
#             assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()

#             assert (calls[1].kwargs["u"] == np.array([1, 2, 3])).all()
#             assert (calls[1].kwargs["v"] == np.array([10, 11, 12])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[0][3], 0.2)
#             assert np.isclose(similarity_matrix[3][0], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])
#             assert math.isnan(similarity_matrix[2][3])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][2])
#             assert math.isnan(similarity_matrix[3][3])

#         @patch("recommenders.common.cosine_similarity")
#         def test_6(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3],
#                                 [4,   5,  6],
#                                 [7,   8,  9],
#                                 [10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((4, 4), np.nan)

#             target = 3

#             neighborhood = np.array([2])
#             compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1

#             assert (calls[0].kwargs["u"] == np.array([10, 11, 12])).all()
#             assert (calls[0].kwargs["v"] == np.array([7, 8, 9])).all()

#             assert np.isclose(similarity_matrix[2][3], 0.1)
#             assert np.isclose(similarity_matrix[3][2], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[0][3])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[3][0])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][3])

#     class TestItemScenario:
#         @patch("recommenders.common.cosine_similarity")
#         def test_1(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 0
#             neighborhood = np.array([1, 2])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2
#             assert (calls[0].kwargs["u"] == np.array([1, 4, 7])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()
#             assert (calls[1].kwargs["u"] == np.array([1, 4, 7])).all()
#             assert (calls[1].kwargs["v"] == np.array([3, 6, 9])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[0][2], 0.2)
#             assert np.isclose(similarity_matrix[2][0], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_2(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 0
#             neighborhood = np.array([1])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1
#             assert (calls[0].kwargs["u"] == np.array([1, 4, 7])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_3(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 2
#             neighborhood = np.array([1])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list

#             assert len(calls) == 1
#             assert (calls[0].kwargs["u"] == np.array([3, 6, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()

#             assert np.isclose(similarity_matrix[1][2], 0.1)
#             assert np.isclose(similarity_matrix[2][1], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_4(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1, 2, 3],
#                                 [4, 5, 6],
#                                 [7, 8, 9]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 2
#             neighborhood = np.array([0])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list

#             assert len(calls) == 1
#             assert (calls[0].kwargs["u"] == np.array([3, 6, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([1, 4, 7])).all()

#             assert np.isclose(similarity_matrix[0][2], 0.1)
#             assert np.isclose(similarity_matrix[2][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_5(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3,  4],
#                                 [5,   6,  7,  8],
#                                 [9,  10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((4, 4), np.nan)
#             target = 0
#             neighborhood = np.array([1, 2, 3])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 3

#             assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 6, 10])).all()

#             assert (calls[1].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[1].kwargs["v"] == np.array([3, 7, 11])).all()

#             assert (calls[2].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[2].kwargs["v"] == np.array([4, 8, 12])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert np.isclose(similarity_matrix[0][2], 0.2)
#             assert np.isclose(similarity_matrix[2][0], 0.2)

#             assert np.isclose(similarity_matrix[0][3], 0.3)
#             assert np.isclose(similarity_matrix[3][0], 0.3)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])
#             assert math.isnan(similarity_matrix[2][3])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][2])
#             assert math.isnan(similarity_matrix[3][3])

#         @patch("recommenders.common.cosine_similarity")
#         def test_6(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3,  4],
#                                 [5,   6,  7,  8],
#                                 [9,  10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#             similarity_matrix = np.full((4, 4), np.nan)
#             target = 0
#             neighborhood = np.array([2])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1

#             assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([3, 7, 11])).all()

#             assert np.isclose(similarity_matrix[0][2], 0.1)
#             assert np.isclose(similarity_matrix[2][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])
#             assert math.isnan(similarity_matrix[2][3])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][2])
#             assert math.isnan(similarity_matrix[3][3])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[0][3])
#             assert math.isnan(similarity_matrix[1][0])

#         @patch("recommenders.common.cosine_similarity")
#         def test_7(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3,  4],
#                                 [5,   6,  7,  8],
#                                 [9,  10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4]

#             similarity_matrix = np.full((4, 4), np.nan)
#             target = 0
#             neighborhood = np.array([3])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1

#             assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([4, 8, 12])).all()

#             assert np.isclose(similarity_matrix[0][3], 0.1)
#             assert np.isclose(similarity_matrix[3][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])
#             assert math.isnan(similarity_matrix[2][3])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][2])
#             assert math.isnan(similarity_matrix[3][3])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[2][0])

#         @patch("recommenders.common.cosine_similarity")
#         def test_8(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3,  4],
#                                 [5,   6,  7,  8],
#                                 [9,  10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2]

#             similarity_matrix = np.full((4, 4), np.nan)
#             target = 0
#             neighborhood = np.array([1])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1

#             assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 6, 10])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[0][3])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[1][3])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])
#             assert math.isnan(similarity_matrix[2][3])
#             assert math.isnan(similarity_matrix[3][0])
#             assert math.isnan(similarity_matrix[3][1])
#             assert math.isnan(similarity_matrix[3][2])
#             assert math.isnan(similarity_matrix[3][3])

#         @patch("recommenders.common.cosine_similarity")
#         def test_9(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3],
#                                 [4,   5,  6],
#                                 [7,   8,  9],
#                                 [10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 0
#             neighborhood = np.array([1])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 1

#             assert (calls[0].kwargs["u"] == np.array([1, 4, 7, 10])).all()
#             assert (calls[0].kwargs["v"] == np.array([2, 5, 8, 11])).all()

#             assert np.isclose(similarity_matrix[0][1], 0.1)
#             assert np.isclose(similarity_matrix[1][0], 0.1)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][2])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[1][2])
#             assert math.isnan(similarity_matrix[2][0])
#             assert math.isnan(similarity_matrix[2][1])
#             assert math.isnan(similarity_matrix[2][2])

#         @patch("recommenders.common.cosine_similarity")
#         def test_10(self, cosine_similarity_mock):
#             # fmt: off
#             dataset = np.array([[1,   2,  3],
#                                 [4,   5,  6],
#                                 [7,   8,  9],
#                                 [10, 11, 12]])
#             # fmt: on

#             cosine_similarity_mock.side_effect = [0.1, 0.2]

#             similarity_matrix = np.full((3, 3), np.nan)
#             target = 2
#             neighborhood = np.array([0, 1])

#             compute_neighborhood_cosine_similarity(
#                 dataset.T, similarity_matrix, target, neighborhood
#             )

#             calls = cosine_similarity_mock.call_args_list
#             assert len(calls) == 2

#             assert (calls[0].kwargs["u"] == np.array([3, 6, 9, 12])).all()
#             assert (calls[0].kwargs["v"] == np.array([1, 4, 7, 10])).all()

#             assert (calls[1].kwargs["u"] == np.array([3, 6, 9, 12])).all()
#             assert (calls[1].kwargs["v"] == np.array([2, 5, 8, 11])).all()

#             assert np.isclose(similarity_matrix[0][2], 0.1)
#             assert np.isclose(similarity_matrix[2][0], 0.1)

#             assert np.isclose(similarity_matrix[1][2], 0.2)
#             assert np.isclose(similarity_matrix[2][1], 0.2)

#             assert math.isnan(similarity_matrix[0][0])
#             assert math.isnan(similarity_matrix[0][1])
#             assert math.isnan(similarity_matrix[1][0])
#             assert math.isnan(similarity_matrix[1][1])
#             assert math.isnan(similarity_matrix[2][2])


# class TestGetSimilarity:

#     def test_1(self):

#         i = 0
#         j = 1

#         dataset = np.array([[1, 2], [3, 4]], dtype=np.float64)
#         similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
#         similarity_strategy = cosine_similarity

#         get_similarity(i, j, dataset, similarity_matrix, similarity_strategy)
#         get_similarity(i, j, dataset)

#         get_similarity(i, j, dataset, similarity_matrix, similarity_strategy)
#         get_similarity(i, j, dataset)

#         # _get_similarity_matrix(dataset)
#         # a = get_similarity_matrix(dataset)


# # @patch("recommenders.common._get_similarity_matrix")
# # def test_get_similarity_matrix(mock):

# #     mock.return_value = np.array([[1, 2], [3, 4]])

# #     assert np.array_equal(
# #         get_similarity_matrix(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]])
# #     )
