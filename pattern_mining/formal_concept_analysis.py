""" formal_concept_analysis.py

This module implements the GreConD algorithm [1] for mining formal concepts from a binary dataset.
It also implements some helper functions to aid in the manipulation of formal concepts.

Copyright 2022 Bernardo C. Rodrigues
See COPYING file for license details

Bibliography
[1] Discovery of optimal factors in binary data via a novel method of matrix decomposition 
    <https://www.sciencedirect.com/science/article/pii/S0022000009000415>
"""

# We shall disable the invalid-name warning for this file because we are using the same variable
# names as in the original paper [1] to make it easier to understand the code.
# pylint: disable=C0103

from collections import namedtuple

import numpy as np
import numba as nb
from typing import List

from dataset.binary_dataset import i, t, _it, assert_binary_dataset

Concept = namedtuple("Concept", "extent intent")


def create_concept(extent, intent):
    """
    Creates a formal concept from the given extent and intent.

    Args:
        extent: A list of row indexes that define the extent of the concept.
        intent: A list of column indexes that define the intent of the concept.

    Returns:
        A Concept object.

    """

    if isinstance(extent, np.ndarray):
        assert extent.dtype == np.int64
        assert extent.ndim == 1
        assert extent.size >= 0
        processed_extent = extent
    else:
        processed_extent = np.asarray(extent, dtype=np.int64)

    if isinstance(intent, np.ndarray):
        assert intent.dtype == np.int64
        assert intent.ndim == 1
        assert intent.size >= 0
        processed_intent = intent
    else:
        processed_intent = np.asarray(intent, dtype=np.int64)

    return Concept(processed_extent, processed_intent)


def concepts_are_equal(concept1: Concept, concept2: Concept) -> bool:
    """
    Checks if two concepts are equal.

    Args:
        concept1: A Concept object.
        concept2: A Concept object.

    Returns:
        True if the concepts are equal, False otherwise.

    """

    assert isinstance(concept1, Concept)
    assert isinstance(concept2, Concept)

    if np.array_equal(concept1.extent, concept2.extent) and np.array_equal(
        concept1.intent, concept2.intent
    ):
        return True
    return False


@nb.njit
def submatrix_intersection_size(
    rows, columns, U
) -> int:  # pragma: no cover # pylint: disable=invalid-name
    """
    Given a submatrix, or bicluster, defined by the rows and columns, this function returns the
    number of True values in the intersection of the submatrix and the matrix U.

    Args:
        rows: A list of row indexes that define the submatrix.
        columns: A list of column indexes that define the submatrix.
        U: The matrix to be intersected with the submatrix.

    Returns:
        The number of True values in the intersection of the submatrix and the matrix U.

    Example:
        U = np.array([[True, False, True], [False, True, True], [True, True, True]])
        rows = [0, 1]
        columns = [0, 2]
        submatrix_intersection_size(rows, columns, U)  # returns 2
    """

    intersection_size = 0
    for row in rows:
        for column in columns:
            if U[row][column]:
                intersection_size += 1
    return intersection_size


@nb.njit
def erase_submatrix_values(
    rows, columns, U
) -> None:  # pragma: no cover # pylint: disable=invalid-name
    """
    Given a submatrix, or bicluster, defined by the rows and columns, this function sets the values
    of the matrix U that are in the intersection of the submatrix and U to False. This effectively
    removes the submatrix from U.

    Args:
        rows: A list of row indexes that define the submatrix.
        columns: A list of column indexes that define the submatrix.
        U: The matrix to be intersected with the submatrix.

    Returns:
        None

    Example:
        U = np.array([[True, False, True], [False, True, True], [True, True, True]])
        rows = [0, 1]
        columns = [0, 2]
        erase_submatrix_values(rows, columns, U)
        U  # returns np.array([[False, False, True], [False, False, True], [True, True, True]])
    """

    for row in rows:
        for column in columns:
            U[row][column] = False


def grecond(binary_dataset: np.ndarray, coverage: float = 1.0) -> tuple[List[Concept], float]:
    """
    Implements Algorithm 2 in section 2.5.2 (page 15) from [1].

    This algorithms proposes a greedy heuristic to enumerate the set F given a binary dataset D.
    F is supposed to be a 'good enough' formal context of D although it's not guaranteed to be
    optimal (smallest F that covers all of D).

    It is also possible to define the desired coverage. The algorithm will stop when the current set
    F covers the given coverage.

    Args:
        binary_dataset: A binary dataset.
        coverage: A float value between 0 and 1 that defines the desired coverage.

    Returns:
        A tuple containing a list of formal concepts and the current coverage.

    """

    assert isinstance(binary_dataset, np.ndarray)
    assert (
        binary_dataset.dtype == bool
        or binary_dataset.dtype == np.bool_
        or binary_dataset.dtype == nb.types.bool_
    )
    assert binary_dataset.ndim == 2
    assert binary_dataset.size > 0

    assert isinstance(coverage, float)
    assert 0 < coverage <= 1

    U = binary_dataset.copy()
    Y = np.arange(binary_dataset.shape[1], dtype=int)
    transposed_binary_dataset = binary_dataset.T

    initial_number_of_trues = np.count_nonzero(U)

    F = []
    current_coverage = 0

    while coverage > current_coverage:
        current_coverage = 1 - np.count_nonzero(U) / initial_number_of_trues
        D = np.array([])
        V = 0
        D_u_j = np.array([])  # current D union {j}

        searching = True
        js_not_in_D = [j for j in Y if j not in D_u_j]

        while searching:
            best_D_u_j_closed_intent = None
            best_D_u_j_V = 0

            for j in js_not_in_D:
                D_u_j = np.append(D, j).astype(int)

                D_u_j_closed_extent = _it(binary_dataset, D_u_j)
                D_u_j_closed_intent = _it(transposed_binary_dataset, D_u_j_closed_extent)

                D_u_j_V = submatrix_intersection_size(D_u_j_closed_extent, D_u_j_closed_intent, U)

                if D_u_j_V > best_D_u_j_V:
                    best_D_u_j_V = D_u_j_V
                    best_D_u_j_closed_intent = D_u_j_closed_intent.copy()

            if best_D_u_j_V > V:
                D = best_D_u_j_closed_intent
                V = best_D_u_j_V
            else:
                searching = False

        C = _it(binary_dataset, D)

        new_concept = create_concept(C, D)

        F.append(new_concept)

        erase_submatrix_values(new_concept.extent, new_concept.intent, U)

        current_coverage = 1 - np.count_nonzero(U) / initial_number_of_trues

    return F, current_coverage


@nb.njit
def _get_matrices(
    concepts: List[Concept], dataset_number_rows: int, dataset_number_cols: int
):  # pragma: no cover
    """
    This function acts as a kernel for the get_factor_matrices_from_concepts function. It is
    implemented in Numba to speed up the process of creating the matrices. It is not meant to be
    called directly. Use the get_factor_matrices_from_concepts function instead.
    """
    Af = np.zeros(shape=(dataset_number_rows, len(concepts)), dtype=nb.bool_)
    Bf = np.zeros(shape=(len(concepts), dataset_number_cols), dtype=nb.bool_)

    for concept_index, concept in enumerate(concepts):
        for item in concept.extent:
            Af[item][concept_index] = True

        for item in concept.intent:
            Bf[concept_index][item] = True

    return Af, Bf


def get_factor_matrices_from_concepts(
    concepts: List[Concept], dataset_number_rows: int, dataset_number_cols: int
):
    """
    This function takes a list of formal concepts and returns two matrices as described in section
    2.1 (page 6) from [1]. If the given formal concepts cover all values from a matrix I,
    I = Af x Bf.

    Args:
        concepts: A list of formal concepts.
        dataset_number_rows: The number of rows in the dataset.
        dataset_number_cols: The number of columns in the dataset.

    Returns:
        Af: A matrix with the same number of rows as the dataset and the same number of columns as
            the number of concepts. Each column represents a formal concept and each row represents
            an object in the dataset. If an object belongs to a concept, the value in the
            corresponding cell will be 1, otherwise it will be 0.
        Bf: A matrix with the same number of rows as the number of concepts and the same number of
            columns as the dataset. Each row represents a formal concept and each column represents
            an attribute in the dataset. If an attribute belongs to a concept, the value in the
            corresponding cell will be 1, otherwise it will be 0.


    Example:
        concepts = [Concept([0, 1], [0, 1]), Concept([0, 1, 2], [0, 1, 2])]
        dataset_number_rows = 4
        dataset_number_cols = 4
        Af, Bf = get_factor_matrices_from_concepts(concepts, dataset_number_rows,
                                                   dataset_number_cols)
        Af  # returns np.array([[1, 1], [1, 1], [0, 1], [0, 1]])
        Bf  # returns np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    """

    # We need to convert the list of concepts to a Numba typed list to be able to use it in the
    # _get_matrices function.

    assert isinstance(concepts, list)
    assert len(concepts) > 0
    assert all(isinstance(concept, Concept) for concept in concepts)
    assert isinstance(dataset_number_rows, int)
    assert dataset_number_rows > 0
    assert isinstance(dataset_number_cols, int)
    assert dataset_number_cols > 0

    typed_concept_list = nb.typed.List()
    for concept in concepts:
        typed_concept_list.append(concept)

    return _get_matrices(typed_concept_list, dataset_number_rows, dataset_number_cols)


def construct_context_from_binaps_patterns(
    binary_dataset: np.ndarray, patterns: List[np.ndarray], closed_itemsets: bool = True
) -> List[Concept]:
    """
    Construct a context from binaps patterns.

    Args:
        binary_dataset: The binary dataset object.
        patterns: A list of binaps patterns represented as lists of integers.
        closed_itemsets: A boolean flag indicating whether to compute closed itemsets
                         (default: True).

    Returns:
        A list of Concept objects representing the constructed context.

    This function constructs a context from the given binaps patterns and the associated binary
    dataset. Each binaps pattern is converted into a tidset and itemset based on the binary dataset.
    The context is represented as a list of Concept objects, where each Concept consists of a tidset
    and an itemset.

    If the `closed_itemsets` flag is set to True, closed itemsets will be computed by transforming
    the itemset into a closed itemset based on the binary dataset.

    Example:
        patterns = [[1, 2, 3], [4, 5], [2, 4, 6]]
        context = construct_context_from_binaps_patterns(binary_dataset, patterns)
    """
    context = []

    assert_binary_dataset(binary_dataset)

    for pattern in patterns:
        max_item = np.max(pattern)
        assert max_item < binary_dataset.shape[1]
        tidset = t(binary_dataset, pattern)  # a pattern equals an itemset

        if closed_itemsets:
            itemset = i(binary_dataset, tidset)
            tidset = t(binary_dataset, itemset)
        else:
            itemset = pattern

        context.append(Concept(tidset, itemset))

    return context
