""" plot.py

This module implements all the functions used to plot the results of the benchmarking process.

Copyright 2023 Bernardo C. Rodrigues
See LICENSE file for license details
"""

import itertools
import multiprocessing
from typing import Tuple, List
from collections import defaultdict
import scipy
import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.graph_objects as go
from surprise import Trainset

from .threads import RecommenderVariation, generic_benchmark_thread

DPI = 300
WIDTH = 1200
HEIGHT = 800
FORMAT = "jpg"

METRIC_NAMES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "fit_time": "Fit Elapsed Time (s)",
    "test_time": "Test Elapsed (s)",
    "precision_at_k": "Precision@20",
    "recall_at_k": "Recall@20",
    "mean_bicluster_size": "Mean Formal Concept Size",
    "mean_bicluster_intent": "Mean Intent",
    "mean_bicluster_extent": "Mean Extent",
}

pd.set_option("display.expand_frame_repr", False)


def customize_default_template():
    """
    Customize the default template with specific layout settings.

    This will give all figures the same look and feel.
    """

    # Access the default template
    default_template = pio.templates[pio.templates.default]

    # Customize font settings
    default_template.layout.font.family = "Latin Modern"
    default_template.layout.font.size = 22
    default_template.layout.font.color = "black"

    # Customize margin and width
    default_template.layout.margin = go.layout.Margin(t=50, b=50, l=50, r=50)
    default_template.layout.width = WIDTH
    default_template.layout.height = HEIGHT

    # Customize background color
    default_template.layout.plot_bgcolor = "rgb(245,245,245)"

    # Customize y-axis settings
    default_template.layout.yaxis = dict(
        mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey"
    )

    # Customize x-axis settings
    default_template.layout.xaxis = dict(
        mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey"
    )

    # Customize legend background color
    default_template.layout.legend = dict(bgcolor="rgb(245,245,245)")

    # Set the default renderer to JPEG
    pio.renderers.default = FORMAT


def coalesce_fold_results(raw_results: List) -> dict:
    """
    Coalesce the raw experiment results into a dictionary that maps recommender variations to
    metrics to folds to lists of results. This function is used to coalesce the results of a
    single fold.

    Args:
        raw_results: List of raw experiment results.
        metric_names: List of metric names to be coalesced.

    Returns:
        Dictionary that maps recommender names to metrics to folds to lists of results.

    Example:
        >>> raw_results = [
        ...     (0, "UBCF", {"mae": 0.1, "rmse": 0.2}),
        ...     (0, "UBCF", {"mae": 0.5, "rmse": 0.7}),
        ...     (1, "UBCF", {"mae": 0.1, "rmse": 0.4}),
        ...     (1, "UBCF", {"mae": 0.6, "rmse": 0.9}),
        ... ]
        >>> coalesce_raw_results(raw_results)
        {
            "UBCF": {
                "mae": {
                    0: [0.1, 0.5],
                    1: [0.1, 0.6],
                },
                "rmse": {
                    0: [0.2, 0.7],
                    1: [0.4, 0.9],
                },
            },
        }

    """
    coalesced_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for raw_experiment_result in raw_results:
        fold, recommender_variation, experiment_results = raw_experiment_result
        for metric_name in experiment_results.keys():
            coalesced_results[recommender_variation][metric_name][fold].append(
                experiment_results[metric_name]
            )
    return coalesced_results


def concatenate_fold_results(coalesced_results: dict) -> dict:
    """
    Concatenate the results of the folds into a single list. This function is used to concatenate
    the results of all folds.

    Args:
        coalesced_results: Dictionary that maps recommender variations to metrics to folds to lists
            of results.

    Returns:
        Dictionary that maps recommender variations to metrics to lists of results.

    Example:
        >>> coalesced_results = {
        ...     "UBCF": {
        ...         "mae": {
        ...             0: [0.1, 0.5],
        ...             1: [0.1, 0.6],
        ...         },
        ...         "rmse": {
        ...             0: [0.2, 0.7],
        ...             1: [0.4, 0.9],
        ...         },
        ...     },
        ... }
        >>> concatenate_fold_results(coalesced_results)
        {
            "UBCF": {
                "mae": [0.1, 0.5, 0.1, 0.6],
                "rmse": [0.2, 0.7, 0.4, 0.9],
            },
        }
    """
    concatenated_results = defaultdict(lambda: defaultdict(list))
    for recommender_name, metric_results in coalesced_results.items():
        for metric_name, fold_results in metric_results.items():
            concatenated_results[recommender_name][metric_name] = list(
                itertools.chain.from_iterable(fold_results.values())
            )
    return concatenated_results


def benchmark(
    folds: List[Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]],
    parallel_recommender_variations: List[RecommenderVariation],
    sequential_recommender_variations: List[RecommenderVariation],
    repeats: int,
    relevance_threshold: float,
    number_of_top_recommendations: int,
    benchmark_thread=generic_benchmark_thread,
    thread_count=multiprocessing.cpu_count(),
):
    """
    Benchmarks a recommender system and returns the raw results.

    Args:
        fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be processed.
            It is a tuple of the fold index and the trainset and testset to be used.
        parallel_recommender_variations (List[RecommenderVariation]): The recommender variations
            that should be evaluated in parallel.
        sequential_recommender_variations (List[RecommenderVariation]): The recommender variations
            that should be evaluated sequentially. This is useful for recommender variations that
            are not thread-safe or that use up a lot of resources.
        repeats (int): The number of times each recommender variation should be evaluated.
        relevance_threshold (float): The threshold that determines whether a rating is relevant or
            not. This is used for calculating metrics that assume a binary prediction (e.g. recall).
        number_of_top_recommendations (int): The number of top recommendations to be considered
            when calculating metrics that assume a top-k list of recommendations (e.g. precision@k).
        benchmark_thread (Callable): The function that should be used to benchmark a single
            recommender variation. It should take the same arguments as the generic_benchmark_thread
            function.
        thread_count (int): The number of threads to be used for parallelization.

    Returns:
        Dictionary that maps recommender variations to metrics to lists of results. The dictionary
        is structured as follows:

        {
            "Recommender Variation 1": {
                "Metric 1": [Result 1, Result 2, Result 3, ...],
                "Metric 2": [Result 1, Result 2, Result 3, ...],
                ...
            },
            "Recommender Variation 2": {
                "Metric 1": [Result 1, Result 2, Result 3, ...],
                "Metric 2": [Result 1, Result 2, Result 3, ...],
                ...
            },
            ...
        }
    """

    assert isinstance(folds, list)
    assert len(folds) > 0
    assert isinstance(parallel_recommender_variations, list)
    assert isinstance(sequential_recommender_variations, list)
    assert isinstance(repeats, int)
    assert repeats > 0
    assert isinstance(relevance_threshold, float)
    assert relevance_threshold > 0.0
    assert isinstance(number_of_top_recommendations, int)
    assert number_of_top_recommendations > 0
    assert callable(benchmark_thread)
    assert isinstance(thread_count, int)
    assert thread_count > 0

    threads_args = list(
        itertools.product(
            folds,
            parallel_recommender_variations,
            [relevance_threshold],
            [number_of_top_recommendations],
        )
    )

    threads_args = repeats * threads_args

    with multiprocessing.Pool(thread_count) as pool:
        raw_experiment_results = pool.starmap(benchmark_thread, iterable=threads_args)

    threads_args = list(
        itertools.product(
            folds,
            sequential_recommender_variations,
            [relevance_threshold],
            [number_of_top_recommendations],
        )
    )

    threads_args = repeats * threads_args

    for thread_args in threads_args:
        raw_experiment_results.append(benchmark_thread(*thread_args))

    coalesced_results = coalesce_fold_results(raw_experiment_results)
    concatenated_results = concatenate_fold_results(coalesced_results)

    return concatenated_results


def calculate_boxplot_values(series: List[float]):
    """
    Calculate boxplot values for a given data set.

    Args:
        series: The data set for which the boxplot values should be calculated.

    Returns:
        q_1: The first quartile.
        q_3: The third quartile.
        lower_fence: The lower fence.
        upper_fence: The upper fence.

    """

    assert isinstance(series, list)
    assert len(series) > 0
    assert all(isinstance(element, float) or isinstance(element, int) for element in series)

    q_1 = np.percentile(series, 25)
    q_3 = np.percentile(series, 75)
    iqr = q_3 - q_1

    lower_fence = q_1 - 1.5 * iqr
    upper_fence = q_3 + 1.5 * iqr

    return q_1, q_3, lower_fence, upper_fence


def plot_metric_box_plot(metric_name: str, concatenated_results: dict):
    """
    Plot a box plot for a given metric.

    Args:
        metric_name: The name of the metric to be plotted.
        concatenated_results: The results as given by concatenate_fold_results.
    """

    fig = go.Figure()
    for recommender_name, metric_results in concatenated_results.items():
        fig.add_trace(
            go.Box(
                y=metric_results[metric_name],
                name=recommender_name,
                fillcolor="gray",
                marker_color="black",
                showlegend=False,  # Add this line to hide the legend
            )
        )

    fig.update_layout(
        yaxis_title=(
            METRIC_NAMES[metric_name] if metric_name in METRIC_NAMES else metric_name.upper()
        ),
        xaxis_title="GreConD coverage 𝐺𝑐𝑜𝑣",
        width=800,
        height=400,
        margin_l=90,
        margin_b=80,
        margin_r=80,
    )

    # margin_l=60,
    # margin_r=100,

    fig.show()
    fig.write_image(f"{metric_name}_boxplot.{FORMAT}", scale=3)


def get_result_table(metric_name: str, concatenated_results: dict):
    """
    Get a table with the results for a given metric.

    Args:
        metric_name: The name of the metric to be plotted.
        concatenated_results: The results as given by concatenate_fold_results.
    """
    results = []
    for recommender_name, metric_results in concatenated_results.items():
        metric_data = metric_results[metric_name]
        mean = np.mean(metric_data)
        standard_deviation = np.std(metric_data)
        min_val = np.min(metric_data)
        max_val = np.max(metric_data)

        results.append(
            {
                "Recommender": recommender_name,
                "Mean": mean,
                "σ": standard_deviation,
                "Min": min_val,
                "Max": max_val,
            }
        )

    return pd.DataFrame(results).to_latex(index=False, float_format="%.3f")


def get_latex_table_from_pandas_table(pandas_table: pd.DataFrame):
    """
    Get a latex table from a pandas table.

    Args:
        pandas_table: The pandas table to be converted to latex.
    """
    return pandas_table.to_latex(index=False, float_format="%.3f")
