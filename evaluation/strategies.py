""" strategies.py

This module contains the implementation of the strategies used to evaluate the
performance of the recommender system. The strategies are divided into two
categories: test and train strategies.

Strategies (strategy design pattern) provide a unified interface to evaluate the
performance of the recommender system.

Client code should use the strategies instead of the standalone functions from
metrics.py.

Copyright 2024 Bernardo C. Rodrigues
See LICENSE file for license details
"""

import statistics

from abc import ABC, abstractmethod
from typing import List, Annotated, Optional
from surprise.accuracy import mae, rmse
from surprise.prediction_algorithms import Prediction, AlgoBase
from pydantic import BaseModel, validate_call, ConfigDict

from annotated_types import Gt

from evaluation.metric import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
    get_f1_score,
    get_ndcg_at_k,
    count_impossible_predictions,
)


class TestMeasureStrategy(ABC, BaseModel):
    """
    Abstract class for test measure strategies.

    Test measurements happen after the model has been trained and tested.
    It should be used to evaluate the model's performance on the prediction list.
    Therefore, measurements of this type will ingest a list of predictions and
    return a float value.

    """

    # This attribute is used to tell pytest to ignore this class as a test case.
    __test__ = False

    @abstractmethod
    def calculate(self, predictions: List[Prediction]) -> float:
        """
        Calculate the measure based on the predictions list.

        Args:
            predictions (List[Prediction]): List of predictions.

        Returns:
            float: The calculated measure value.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Returns the name of the measure.
        """

    @staticmethod
    @abstractmethod
    def is_better_higher() -> bool:
        """
        Returns whether a higher value of the measure is better.

        This is useful for the evaluation process in order to compare and rank
        different models based on the measure value.

        Returns:
            bool: True if higher values are better, False otherwise.
        """

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def __call__(self, predictions: List[Prediction]) -> float:
        """
        Calculate the measure based on the predictions list.

        Args:
            predictions (List[Prediction]): List of predictions.

        Returns:
            float: The calculated measure value.
        """
        return self.calculate(predictions)


class TrainMeasureStrategy(ABC, BaseModel):
    """
    Abstract class for train measure strategies.

    Train measurements happen after the model has been trained and are based on
    trained model. It should be used to evaluate the model's characteristics
    and perhaps its performance on the training data. Therefore, measurements
    of this type will ingest the model and return a float value.
    """

    @abstractmethod
    def calculate(self, recommender_system: AlgoBase) -> float:
        """
        Calculate the measure based on the recommender system.

        Args:
            recommender_system (AlgoBase): The trained recommender system.

        Returns:
            float: The calculated measure value.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Returns the name of the measure.
        """

    @staticmethod
    @abstractmethod
    def is_better_higher() -> bool:
        """
        Returns whether a higher value of the measure is better.

        This is useful for the evaluation process in order to compare and rank
        different models based on the measure value.

        Returns:
            bool: True if higher values are better, False otherwise.
        """

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def __call__(self, recommender_system: AlgoBase) -> float:
        """
        Calculate the measure based on the recommender system.

        Args:
            recommender_system (AlgoBase): The trained recommender system.

        Returns:
            float: The calculated measure value.
        """
        return self.calculate(recommender_system)


class MAEStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    verbose: bool = False

    @staticmethod
    def get_name() -> str:
        return "mae"

    @staticmethod
    def is_better_higher() -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return mae(resolved_predictions, verbose=self.verbose)


class RMSEStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    verbose: bool = False

    @staticmethod
    def get_name() -> str:
        return "rmse"

    @staticmethod
    def is_better_higher() -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return rmse(resolved_predictions, verbose=self.verbose)


class MicroAveragedRecallStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "micro_averaged_recall"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_micro_averaged_recall(resolved_predictions, threshold=self.threshold)


class MacroAveragedRecallStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "macro_averaged_recall"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_macro_averaged_recall(resolved_predictions, threshold=self.threshold)


class RecallAtKStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    k: Annotated[int, Gt(0)] = 10
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "recall_at_k"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_recall_at_k(resolved_predictions, k=self.k, threshold=self.threshold)


class MicroAveragedPrecisionStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "micro_averaged_precision"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_micro_averaged_precision(resolved_predictions, threshold=self.threshold)


class MacroAveragedPrecisionStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "macro_averaged_precision"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_macro_averaged_precision(resolved_predictions, threshold=self.threshold)


class PrecisionAtKStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    k: Annotated[int, Gt(0)] = 10
    threshold: Annotated[float, Gt(0.0)] = 1.0

    @staticmethod
    def get_name() -> str:
        return "precision_at_k"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_precision_at_k(resolved_predictions, k=self.k, threshold=self.threshold)


class F1ScoreStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    threshold: Annotated[float, Gt(0.0)] = 1.0
    k: Optional[Annotated[int, Gt(0)]] = None

    @staticmethod
    def get_name() -> str:
        return "f1_score"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_f1_score(resolved_predictions, threshold=self.threshold, k=self.k)


class CountImpossiblePredictionsStrategy(TestMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "count_impossible_predictions"

    @staticmethod
    def is_better_higher() -> bool:
        return False

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return count_impossible_predictions(predictions)


class PredictionCoverageStrategy(TestMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "prediction_coverage"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:
        return 1 - (count_impossible_predictions(predictions) / len(predictions))


class NDCGStrategy(TestMeasureStrategy):

    include_impossible_predictions: bool = False
    k: Annotated[int, Gt(0)] = 10

    @staticmethod
    def get_name() -> str:
        return "nDCG_at_k"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, predictions: List[Prediction]) -> float:

        if self.include_impossible_predictions:
            resolved_predictions = predictions
        else:
            resolved_predictions = [
                prediction
                for prediction in predictions
                if prediction.details["was_impossible"] is False
            ]

        return get_ndcg_at_k(resolved_predictions, k=self.k)


class BiclusteringCoverageStrategy(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "biclustering_coverage"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:
        return recommender_system.mining_strategy.actual_coverage


class BiclusterCountStrategy(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "bicluster_count"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:
        
        if len(recommender_system.biclusters) == 0:
            raise ValueError("No biclusters found.")
        
        return len(recommender_system.biclusters)


class MeanBiclusterSizeStrategy(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "mean_bicluster_size"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        if len(recommender_system.biclusters) == 0:
            raise ValueError("No biclusters found.")

        mean_bicluster_size = statistics.mean(
            [
                len(bicluster.extent) * len(bicluster.intent)
                for bicluster in recommender_system.biclusters
            ]
        )
        return mean_bicluster_size


class MeanBiclusterIntentStrategy(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "mean_bicluster_intent"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        if len(recommender_system.biclusters) == 0:
            raise ValueError("No biclusters found.")

        mean_bicluster_intent = statistics.mean(
            [len(bicluster.intent) for bicluster in recommender_system.biclusters]
        )
        return mean_bicluster_intent


class MeanBiclusterExtentStrategy(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "mean_bicluster_extent"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:

        if len(recommender_system.biclusters) == 0:
            raise ValueError("No biclusters found.")

        mean_bicluster_extent = statistics.mean(
            [len(bicluster.extent) for bicluster in recommender_system.biclusters]
        )
        return mean_bicluster_extent


class ItemCoverage(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "item_coverage"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:
        return recommender_system.item_coverage


class UserCoverage(TrainMeasureStrategy):

    @staticmethod
    def get_name() -> str:
        return "user_coverage"

    @staticmethod
    def is_better_higher() -> bool:
        return True

    @validate_call(
        config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True)
    )
    def calculate(self, recommender_system: AlgoBase) -> float:
        return recommender_system.user_coverage
