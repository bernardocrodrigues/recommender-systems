""" 
common.py

This module contains common functions used by the recommenders.
"""

from typing import List, Callable

import numpy as np
import numba as nb


from pattern_mining.formal_concept_analysis import Concept as Bicluster


@nb.njit()
def cosine_similarity(
    dataset: np.ndarray, u_id: int, v_id: int, means: np.ndarray, eps: float = 1e-08
) -> float:
    """
    Computes the cosine similarity between two vectors u and v. The cosine similarity is defined
    as follows:

    cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)
                            = sum(u[i] * v[i]) / (sqrt(sum(u[i] ** 2)) * sqrt(sum(v[i] ** 2))),
                              for all i in [0, n)

        where u and v are two vectors and ||u|| is the norm of u and n is the size of the vectors.

    Unlike scipy.spatial.distance.cosine, this function handles NaN values in the vectors. If a
    NaN value is found in the vectors, that coordinate is ignored in the calculation of the
    similarity. If all coordinates are NaN, the similarity is NaN. In addition, this function
    returns the similarity instead of the dissimilarity.

    Args:
        u (np.ndarray): The first vector.
        v (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between u and v.

    Raises:
        AssertionError: If the vectors are not 1D numpy arrays.
        AssertionError: If the vectors have different sizes.
        AssertionError: If the vectors are not of type np.float64.
    """

    u = dataset[u_id]
    v = dataset[v_id]

    not_null_u = np.nonzero(~np.isnan(u))[0]
    not_null_v = np.nonzero(~np.isnan(v))[0]

    common_indices_in_uv = np.intersect1d(not_null_u, not_null_v)

    if common_indices_in_uv.size == 0:
        return np.NaN

    common_u = u[common_indices_in_uv]
    common_v = v[common_indices_in_uv]

    numerator = np.dot(common_u, common_v)

    uu = np.dot(common_u, common_u)
    vv = np.dot(common_v, common_v)

    denominator = max(np.sqrt(uu * vv), eps)

    similarity = numerator / denominator

    return similarity


@nb.njit()
def pearson_similarity(
    dataset: np.ndarray, u_id: int, v_id: int, means: np.ndarray, eps: float = 1e-08
) -> float:

    u = dataset[u_id]
    v = dataset[v_id]
    mean_u = means[u_id]
    mean_v = means[v_id]

    not_null_u = np.nonzero(~np.isnan(u))[0]
    not_null_v = np.nonzero(~np.isnan(v))[0]

    common_indices_in_uv = np.intersect1d(not_null_u, not_null_v)

    if common_indices_in_uv.size == 0:
        return np.NaN

    common_u = u[common_indices_in_uv]
    common_v = v[common_indices_in_uv]

    common_u = common_u - mean_u
    common_v = common_v - mean_v

    numerator = np.dot(common_u, common_v)

    uu = np.dot(common_u, common_u)
    vv = np.dot(common_v, common_v)

    denominator = max(np.sqrt(uu * vv), eps)

    similarity = numerator / denominator

    return similarity


@nb.njit()
def adjusted_cosine_similarity(
    dataset: np.ndarray, i_id: int, j_id: int, means: np.ndarray, eps: float = 1e-08
) -> float:

    i = dataset[i_id]
    j = dataset[j_id]

    not_null_i = np.nonzero(~np.isnan(i))[0]
    not_null_j = np.nonzero(~np.isnan(j))[0]

    common_indices_in_ij = np.intersect1d(not_null_i, not_null_j)

    if common_indices_in_ij.size == 0:
        return np.NaN

    common_i = i[common_indices_in_ij]
    common_j = j[common_indices_in_ij]

    common_i = common_i - means[common_indices_in_ij]
    common_j = common_j - means[common_indices_in_ij]

    numerator = np.dot(common_i, common_j)

    ii = np.dot(common_i, common_i)
    jj = np.dot(common_j, common_j)

    denominator = max(np.sqrt(ii * jj), eps)

    similarity = numerator / denominator

    return similarity


@nb.njit()
def user_pattern_similarity(user: np.ndarray, pattern: Bicluster) -> float:
    """
    Calculates the similarity between a user and a pattern (bicluster) based on the number of items
    they have in common. The similarity is defined as follows:

            similarity = |I_u ∩ I_p| / |I_u ∩ I_p| + |I_p - I_u|
                       = |I_u ∩ I_p| / |I_p|

        where I_u is the set of relevant items for the user and I_p is the set of items
        for the pattern.

    This similarity metric is used is defined by Symeonidis[1].

    [1] Symeonidis, P., Nanopoulos, A., Papadopoulos, A., & Manolopoulos, Y. (n.d.).
        Nearest-Biclusters Collaborative Filtering with Constant Values. Lecture Notes in Computer
        Science, 36-55. doi:10.1007/978-3-540-77485-3

    Args:
        user (np.ndarray): A 1D numpy array representing the user's relevant items. The array must
        be an itemset representation.
        pattern (Bicluster): A Bicluster representing the pattern.

    Returns:
        float: A value between 0 and 1 representing the similarity between the user and the pattern.
    """

    number_of_itens_from_pattern_in_user = np.intersect1d(user, pattern.intent).size

    if pattern.intent.size == 0:
        return 0.0

    similarity = number_of_itens_from_pattern_in_user / pattern.intent.size

    return similarity


@nb.njit()
def weight_frequency(user: np.ndarray, pattern: Bicluster) -> float:
    """
    Calculates the similarity between a user and a pattern (bicluster) based on the number of items
    they have in common and the number of users in the pattern. The similarity is defined as follows:

            similarity = |U_p| * (|I_u ∩ I_p| / |I_p|)

        where U_p is the set of users in the pattern, I_u is the set of relevant items for the user
        and I_p is the set of items for the pattern.

    This similarity metric is used is defined by Symeonidis[1].

    [1] Symeonidis, P., Nanopoulos, A., Papadopoulos, A., & Manolopoulos, Y. (n.d.).
        Nearest-Biclusters Collaborative Filtering with Constant Values. Lecture Notes in Computer
        Science, 36-55. doi:10.1007/978-3-540-77485-3

    Args:
        user (np.ndarray): A 1D numpy array representing the user's relevant items. The array must
        be an itemset representation.
        pattern (Bicluster): A Bicluster representing the pattern.

    Returns:
        float: A value between 0 and 1 representing the similarity between the user and the pattern.

    """

    number_of_users_in_pattern = pattern.extent.size

    return number_of_users_in_pattern * user_pattern_similarity(user, pattern)


@nb.njit()
def double_weight_frequency(user: np.ndarray, pattern: Bicluster) -> float:
    """
    Calculates the similarity between a user and a pattern (bicluster) based on the number of items
    they have in common and the number of users and items in the pattern. The similarity is defined
    as follows:

            similarity = |U_p| * |I_p| * (|I_u ∩ I_p| / |I_p|)

        where U_p is the set of users in the pattern, I_u is the set of relevant items for the user
        and I_p is the set of items for the pattern.
    """

    number_of_users_in_pattern = pattern.extent.size
    number_of_items_in_pattern = pattern.intent.size

    return (
        number_of_users_in_pattern
        * number_of_items_in_pattern
        * user_pattern_similarity(user, pattern)
    )


@nb.njit()
def get_similarity(
    i: int,
    j: int,
    dataset: np.ndarray,
    means: np.ndarray,
    similarity_matrix: np.ndarray = None,
    similarity_strategy: Callable = pearson_similarity,
) -> float:
    """
    Given a np.ndarray and some method that calculates some distance between two vector,
    computes the similarity between two users (rows).

    If a similarity matrix is provided, the function will check if the similarity between the two
    users has already been calculated. If so, the function will return the similarity stored in the
    matrix. Otherwise, the function will calculate the similarity between the two users and store
    it in the matrix.
    """

    if similarity_matrix is not None and not np.isnan(similarity_matrix[i, j]):
        return similarity_matrix[i, j]
    if i == j:
        similarity = 1.0
    else:
        similarity = similarity_strategy(dataset, i, j, means)

    if similarity_matrix is not None:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    return similarity


@nb.njit()
def get_similarity_matrix(
    dataset: np.ndarray, means: np.ndarray, similarity_strategy=pearson_similarity
):
    """
    Given a np.ndarray and some method that calculates some distance between two vector,
    computes the similarity matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1
    implies that the vectors are completely different (maximum distance) while a return value of 0
    implies that the vectors are identical (minimum distance).
    """
    similarity_matrix = np.full((dataset.shape[0], dataset.shape[0]), np.NaN, dtype=np.float64)

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if j >= i:
                get_similarity(i, j, dataset, means, similarity_matrix, similarity_strategy)

    return similarity_matrix


@nb.njit()
def get_top_k_biclusters_for_user(
    biclusters: List[Bicluster],
    user_as_tidset: np.ndarray,
    number_of_top_k_patterns: int,
    similarity_strategy: Callable = user_pattern_similarity,
) -> List[Bicluster]:
    """
    Gets the top-k patterns for a given user. The top-k patterns are the patterns that
    have the highest similarity with the user.

    Args:
        patterns (List[Bicluster]): The patterns that will be analyzed. Each pattern must be an
                                    itemset representation. This list cannot be empty.
        user_as_tidset (np.ndarray): The target user. The array must be an tidset representation.
        number_of_top_k_patterns (int): The number of patterns to return.

    Returns:
        List[Bicluster]: The top-k patterns. The patterns are sorted in ascending order of
                        similarity.
    """

    similar_biclusters = nb.typed.List()
    similarities = nb.typed.List()

    for bicluster in biclusters:
        similarity = similarity_strategy(user_as_tidset, bicluster)
        if similarity > 0:
            similar_biclusters.append(bicluster)
            similarities.append(similarity)

    sorted_similarities_indexes = [
        x for x, y in sorted(enumerate(similarities), key=lambda x: x[1])
    ]
    sorted_similar_patterns = [similar_biclusters[i] for i in sorted_similarities_indexes]
    top_k_patterns = sorted_similar_patterns[-number_of_top_k_patterns:]

    return top_k_patterns


@nb.njit()
def get_indices_above_threshold(subset: np.ndarray, binarization_threshold: float) -> np.ndarray:
    """
    Gets the indices of the elements in a subset that are above a given threshold. If this subset
    is a row or column of a matrix, this function can be used to get the user in a tidset
    representation or the items in an itemset representation, respectively.

    Args:
        subset (np.ndarray): A row or column of a matrix.
        binarization_threshold (float): The threshold. Elements above this threshold will be
                                        considered relevant and will be present in the returned
                                        indices.

    Returns:
        np.ndarray: The indices of the elements in the subset that are above the threshold.
    """

    binarized_subset = subset >= binarization_threshold
    indices_above_threshold = np.nonzero(binarized_subset)[0]
    return indices_above_threshold


@nb.njit()
def merge_biclusters(
    biclusters: List[Bicluster],
) -> Bicluster:
    """
    Merges a list of biclusters into a single bicluster. This means that the extent of the new
    bicluster will be the union of the extents of the given biclusters and the intent of the new
    bicluster will be the union of the intents of the given biclusters.

    Args:
        biclusters (List[Bicluster]): A list of biclusters.

    Returns:
        Concept: A new bicluster that is the result of merging the given biclusters.
    """

    new_bicluster_extent = np.empty(0, dtype=np.int64)
    new_bicluster_intent = np.empty(0, dtype=np.int64)

    for bicluster in biclusters:
        new_bicluster_extent = np.union1d(new_bicluster_extent, bicluster.extent)
        new_bicluster_intent = np.union1d(new_bicluster_intent, bicluster.intent)

    return Bicluster(extent=new_bicluster_extent, intent=new_bicluster_intent)
