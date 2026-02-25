import abc
import logging
import operator
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
import scores
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc, derived, utils

logger = logging.getLogger(__name__)
CLIMO_T2M_PATH = Path("/huge/users/criedel/ACC-CLIMO/keep/CLIMO_T2M.nc")


class ComputeDocstringMetaclass(abc.ABCMeta):
    """A metaclass that maps the docstring from self._compute_metric() to
    self.compute_metric().

    The `BaseMetric` abstract base class requires users to override a function called
    `_compute_metric()`, while providing a standardized public interface to this method
    called `compute_metric()`. This metaclass automatically maps the docstring from
    `_compute_metric()` to `compute_metric()` so that the documentation a user provides
    for their implementation will automatically appear with the public interface without
    any additional effort.
    """

    def __new__(cls, name, bases, namespace):
        cls = super().__new__(cls, name, bases, namespace)
        # NOTE: the `compute_metric()` method will be defined in the ABC `BaseMetric`,
        # and we never expect the user re-implement it. So it won't be in the namespace
        # of the concrete metric classes - it will only be in the namespace of the ABC
        # `BaseMetric`, and will be available as an attribute of the concrete metric
        # classes.
        if "_compute_metric" in namespace and hasattr(cls, "compute_metric"):
            # Transfer the docstring from _compute_metric to compute_metric, if the
            # former exists.
            if cls._compute_metric.__doc__ is not None:
                # Create a new method for _this_ class, so we can avoid overwriting what
                # we set for the parent.
                _original_compute_metric = cls.compute_metric

                def _compute_metric_with_docstring(self, *args, **kwargs):
                    return _original_compute_metric(self, *args, **kwargs)

                _compute_metric_with_docstring.__doc__ = cls._compute_metric.__doc__
                cls.compute_metric = _compute_metric_with_docstring

        return cls


class BaseMetric(abc.ABC, metaclass=ComputeDocstringMetaclass):
    """Abstract base class defining the foundational interface for all metrics.

    Metrics are general operations applied between forecast and analysis xarray
    DataArrays. EWB metrics prioritize the use of any arbitrary sets of
    forecasts and analyses, so long as the spatiotemporal dimensions are the
    same.

    Public methods:
        compute_metric: Public interface to compute the metric
        maybe_expand_composite: Expand composite metrics into individual metrics
        is_composite: Check if this is a composite metric
        __repr__: String representation of the metric
        __eq__: Check equality with another metric

    Abstract methods:
        _compute_metric: Logic to compute the metric (must be implemented)
    """

    def __init__(
        self,
        name: str,
        preserve_dims: str = "lead_time",
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
    ):
        """Initialize the base metric.

        Args:
            name: The name of the metric.
            preserve_dims: The dimensions to preserve in the computation.
                Defaults to "lead_time".
            forecast_variable: The forecast variable to use in the
                computation.
            target_variable: The target variable to use in the computation.
        """
        # Store the original variables (str or DerivedVariable instances)
        # Do NOT convert to string to preserve output_variables info
        self.name = name
        self.preserve_dims = preserve_dims
        self.forecast_variable = forecast_variable
        self.target_variable = target_variable
        # Check if both variables are None - this is allowed
        if self.forecast_variable is None and self.target_variable is None:
            pass
        # If only one is None, raise an error
        elif self.forecast_variable is None or self.target_variable is None:
            raise ValueError(
                "Both forecast_variable and target_variable must be provided, "
                "or both must be None"
            )

    @abc.abstractmethod
    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        All implementations must accept **kwargs to handle extra
        parameters gracefully, even if they don't use them.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional parameters. Common ones include preserve_dims
                (dimension(s) to preserve, defaults to "lead_time").

        Returns:
            The computed metric result.
        """
        pass

    def compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Public interface to compute the metric.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments to pass to the
                metric implementation.

        Returns:
            The computed metric result.
        """

        # If the forecast or target is sparse, densify it.
        # Ideally we would keep the sparse data structure, but Dask and sparse
        # do not play well together as of Nov 2025.
        forecast = utils.maybe_densify_dataarray(forecast)
        target = utils.maybe_densify_dataarray(target)
        return self._compute_metric(forecast, target, **kwargs)

    def maybe_expand_composite(self) -> Sequence["BaseMetric"]:
        """Expand composite metrics into individual metrics.

        Base implementation returns [self]. Override for composites.

        Returns:
            List containing just this metric.
        """
        return [self]

    def is_composite(self) -> bool:
        """Check if this is a composite metric.

        Base implementation returns False. Override for composites.

        Returns:
            False for base metrics.
        """
        return False

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for metric evaluation.

        Base implementation just returns kwargs as-is.
        Override for metrics that need special preparation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

        Returns:
            Dictionary of kwargs (unchanged for base metrics).
        """
        return base_kwargs.copy()


class CompositeMetric(BaseMetric):
    """Base class for composite metrics that can contain multiple sub-metrics.

    Extends BaseMetric to provide functionality for composite metrics that
    aggregate multiple individual metrics for efficient evaluation.

    Public methods:
        maybe_expand_composite: Expand into individual metrics (overrides base)
        is_composite: Check if has sub-metrics (overrides base)

    Abstract methods:
        maybe_prepare_composite_kwargs: Prepare kwargs for composite evaluation
        _compute_metric: Compute the metric (must be implemented by subclasses)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the composite metric.

        Args:
            *args: Positional arguments passed to BaseMetric.__init__
            **kwargs: Keyword arguments passed to BaseMetric.__init__
        """
        super().__init__(*args, **kwargs)
        self._metric_instances: list["BaseMetric"] = []

    def maybe_expand_composite(self) -> Sequence["BaseMetric"]:
        """Expand composite metrics into individual metrics.

        Returns:
            List containing just this metric.
        """
        if self._metric_instances:
            return self._metric_instances
        return [self]

    def is_composite(self) -> bool:
        """Check if this is a composite metric.

        Returns:
            True if composite (has sub-metrics), False otherwise.
        """
        return bool(self._metric_instances)

    @abc.abstractmethod
    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Returns:
            Dictionary of kwargs (unchanged for composite metrics).
        """

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute metric (not supported for CompositeMetric base).

        CompositeMetric must be subclassed (like ThresholdMetric, LandfallMetric)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "CompositeMetric._compute_metric must be implemented "
            "by subclasses (ThresholdMetric, LandfallMetric) or use "
            "CompositeMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class ThresholdMetric(CompositeMetric):
    """Base class for threshold-based metrics with binary classification.

    Extends CompositeMetric to provide functionality for metrics that require
    forecast and target thresholds for binarization. Can be used as a base
    class for specific threshold metrics or as a composite metric.

    Public methods:
        transformed_contingency_manager: Create contingency manager
        maybe_prepare_composite_kwargs: Prepare kwargs (overrides parent)
        __call__: Make instances callable with configured thresholds

    Abstract methods:
        _compute_metric: Compute the metric (must be implemented by subclasses)

    Usage patterns:
        1. As a base class for specific metrics (CriticalSuccessIndex, etc.)
        2. As a composite metric to compute multiple threshold metrics
           efficiently by reusing the transformed contingency manager

    Example:
        composite = ThresholdMetric(
            metrics=[CriticalSuccessIndex, FalseAlarmRatio, Accuracy],
            forecast_threshold=0.7,
            target_threshold=0.5
        )
    """

    def __init__(
        self,
        name: str = "threshold_metrics",
        preserve_dims: str = "lead_time",
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        metrics: Optional[list[Type["ThresholdMetric"]]] = None,
        **kwargs,
    ):
        """Initialize the threshold metric.

        Args:
            name: The name of the metric. Defaults to "threshold_metrics".
            preserve_dims: The dimensions to preserve in the computation.
                Defaults to "lead_time".
            forecast_variable: The forecast variable to use in the
                computation.
            target_variable: The target variable to use in the computation.
            forecast_threshold: The threshold for binarizing the forecast.
                Defaults to 0.5.
            target_threshold: The threshold for binarizing the target.
                Defaults to 0.5.
            metrics: A list of metrics to use as a composite. Defaults to
                None.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(
            name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            **kwargs,
        )
        self.forecast_threshold = forecast_threshold
        self.target_threshold = target_threshold
        self.preserve_dims = preserve_dims
        self.metrics = metrics or []

        # If metrics provided, instantiate them
        if self.metrics is not None:
            self._metric_instances = [
                (
                    metric_cls(
                        forecast_threshold=self.forecast_threshold,
                        target_threshold=self.target_threshold,
                        preserve_dims=self.preserve_dims,
                    )
                    if isinstance(metric_cls, type)
                    else metric_cls
                )
                for metric_cls in self.metrics
            ]
        else:
            self._metric_instances = []

    def __call__(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs) -> Any:
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the instance method with the configured parameters
        return self.compute_metric(forecast, target, **kwargs)

    def transformed_contingency_manager(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        forecast_threshold: float,
        target_threshold: float,
        preserve_dims: str,
        op_func: Union[
            Callable, Literal[">", ">=", "<", "<=", "==", "!="]
        ] = operator.ge,
    ) -> scores.categorical.BasicContingencyManager:
        """Create and transform a contingency manager.

        This method is used to create and transform a contingency manager from the
        scores module. The op_func is used to binarize the forecast and target data with
        either a string representation of the operator, e.g. ">=", or a callable
        function from the operator module, e.g. operator.ge.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            forecast_threshold: Threshold for binarizing forecast.
            target_threshold: Threshold for binarizing target.
            preserve_dims: Dimension(s) to preserve during transform.
            op_func: Function or string representation of the operator to apply to the
                forecast and target. Defaults to operator.ge (greater than or equal to).

        Returns:
            Transformed contingency manager.
        """
        # Apply thresholds to binarize the data
        op_func = utils.maybe_get_operator(op_func)
        binary_forecast = utils.maybe_densify_dataarray(
            op_func(forecast, forecast_threshold)
        ).astype(float)
        binary_target = utils.maybe_densify_dataarray(
            op_func(target, target_threshold)
        ).astype(float)

        # Create and transform contingency manager
        binary_contingency_manager = scores.categorical.BinaryContingencyManager(
            binary_forecast, binary_target
        )
        transformed = binary_contingency_manager.transform(preserve_dims=preserve_dims)

        return transformed

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Computes the transformed contingency manager once and adds
        it to kwargs for efficient composite evaluation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

        Returns:
            Dictionary of kwargs including transformed_manager.
        """
        kwargs = base_kwargs.copy()

        if self.is_composite() and len(self._metric_instances) > 1:
            kwargs["transformed_manager"] = self.transformed_contingency_manager(
                forecast=forecast_data,
                target=target_data,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
            kwargs["forecast_threshold"] = self.forecast_threshold
            kwargs["target_threshold"] = self.target_threshold
            kwargs["preserve_dims"] = self.preserve_dims

        return kwargs

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute metric (not supported for ThresholdMetric base).

        ThresholdMetric must be subclassed (like CriticalSuccessIndex, FalseAlarmRatio)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "ThresholdMetric._compute_metric must be implemented "
            "by subclasses (CriticalSuccessIndex, FalseAlarmRatio, etc.) or use "
            "ThresholdMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class CriticalSuccessIndex(ThresholdMetric):
    """Compute Critical Success Index (CSI) from binary classifications.

    Extends ThresholdMetric to compute CSI between forecast and target using
    the preserve_dims dimensions. CSI measures the fraction of correctly
    predicted events.
    """

    def __init__(self, name: str = "CriticalSuccessIndex", *args, **kwargs):
        """Initialize the Critical Success Index metric.

        Args:
            name: The name of the metric. Defaults to
                "CriticalSuccessIndex".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.critical_success_index()


class FalseAlarmRatio(ThresholdMetric):
    """Compute False Alarm Ratio (FAR) from binary classifications.

    Extends ThresholdMetric to compute FAR between forecast and target using
    the preserve_dims dimensions. FAR measures the fraction of predicted
    events that did not occur. Note: FAR is not the same as False Alarm Rate.
    """

    def __init__(self, name: str = "FalseAlarmRatio", *args, **kwargs):
        """Initialize the False Alarm Ratio metric.

        Args:
            name: The name of the metric. Defaults to "FalseAlarmRatio".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.false_alarm_ratio()


class TruePositives(ThresholdMetric):
    """Compute True Positive ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of true positives (correctly
    predicted events) to the total number of observations. Corresponds to the
    top right cell in the contingency table.
    """

    def __init__(self, name: str = "TruePositives", *args, **kwargs):
        """Initialize the True Positives metric.

        Args:
            name: The name of the metric. Defaults to "TruePositives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["tp_count"] / counts["total_count"]


class FalsePositives(ThresholdMetric):
    """Compute False Positive ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of false positives
    (incorrectly predicted events) to the total number of observations.
    """

    def __init__(self, name: str = "FalsePositives", *args, **kwargs):
        """Initialize the False Positives metric.

        Args:
            name: The name of the metric. Defaults to "FalsePositives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["fp_count"] / counts["total_count"]


class TrueNegatives(ThresholdMetric):
    """Compute True Negative ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of true negatives (correctly
    predicted non-events) to the total number of observations.
    """

    def __init__(self, name: str = "TrueNegatives", *args, **kwargs):
        """Initialize the True Negatives metric.

        Args:
            name: The name of the metric. Defaults to "TrueNegatives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["tn_count"] / counts["total_count"]


class FalseNegatives(ThresholdMetric):
    """Compute False Negative ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of false negatives (missed
    events) to the total number of observations. Corresponds to the top left
    cell in the contingency table.
    """

    def __init__(self, name: str = "FalseNegatives", *args, **kwargs):
        """Initialize the False Negatives metric.

        Args:
            name: The name of the metric. Defaults to "FalseNegatives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["fn_count"] / counts["total_count"]


class Accuracy(ThresholdMetric):
    """Compute classification accuracy from binary classifications.

    Extends ThresholdMetric to compute the ratio of correct predictions (true
    positives + true negatives) to the total number of observations. Measures
    overall correctness of the forecast.
    """

    def __init__(self, name: str = "Accuracy", *args, **kwargs):
        """Initialize the Accuracy metric.

        Args:
            name: The name of the metric. Defaults to "Accuracy".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.accuracy()


class FrequencyBias(ThresholdMetric):
    """Compute Frequency Bias from binary classifications.

    Extends ThresholdMetric to compute the frequency bias (also called bias
    score) between forecast and target. Frequency bias = (hits + false_alarms)
    / (hits + misses), or equivalently the ratio of forecast yes events to
    observed yes events. A value of 1.0 indicates no bias; >1 means
    over-prediction; <1 means under-prediction.
    """

    def __init__(self, name: str = "FrequencyBias", *args, **kwargs):
        """Initialize the Frequency Bias metric.

        Args:
            name: The name of the metric. Defaults to "FrequencyBias".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.frequency_bias()


class EquitableThreatScore(ThresholdMetric):
    """Compute Equitable Threat Score (ETS / Gilbert Skill Score).

    Extends ThresholdMetric to compute ETS, which accounts for hits due to
    random chance. ETS = (hits - hits_random) / (hits + misses + false_alarms
    - hits_random). Range: -1/3 to 1, where 0 = no skill, 1 = perfect.
    """

    def __init__(self, name: str = "EquitableThreatScore", *args, **kwargs):
        """Initialize the Equitable Threat Score metric.

        Args:
            name: The name of the metric. Defaults to "EquitableThreatScore".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.equitable_threat_score()


class MeanSquaredError(BaseMetric):
    """Compute Mean Squared Error between forecast and target.

    Extends BaseMetric to calculate MSE with optional interval-based
    weighting and custom weights for spatial/temporal averaging.
    """

    def __init__(
        self,
        name: str = "MeanSquaredError",
        interval_where_one: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        interval_where_positive: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        weights: Optional[xr.DataArray] = None,
        *args,
        **kwargs,
    ):
        """Initialize the Mean Squared Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanSquaredError".
            interval_where_one: Endpoints of the interval where threshold
                weights are 1. Must be increasing. Infinite endpoints
                permissible.
            interval_where_positive: Endpoints of the interval where threshold
                weights are positive. Must be increasing.
            weights: Array of weights to apply to the score (e.g., latitude
                weighting). If None, no weights are applied.
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, *args, **kwargs)
        self.interval_where_one = interval_where_one
        self.interval_where_positive = interval_where_positive
        self.weights = weights

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        if self.interval_where_one is not None:
            return scores.continuous.tw_squared_error(
                forecast,
                target,
                interval_where_one=self.interval_where_one,
                interval_where_positive=self.interval_where_positive,
                weights=self.weights,
                preserve_dims=self.preserve_dims,
            )
        return scores.continuous.mse(forecast, target, preserve_dims=self.preserve_dims)


class MeanAbsoluteError(BaseMetric):
    """Compute Mean Absolute Error between forecast and target.

    Extends BaseMetric to calculate MAE with optional interval-based
    weighting and custom weights for spatial/temporal averaging.
    """

    def __init__(
        self,
        name: str = "MeanAbsoluteError",
        interval_where_one: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        interval_where_positive: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        weights: Optional[xr.DataArray] = None,
        *args,
        **kwargs,
    ):
        """Initialize the Mean Absolute Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanAbsoluteError".
            interval_where_one: Endpoints of the interval where threshold
                weights are 1. Must be increasing. Infinite endpoints
                permissible.
            interval_where_positive: Endpoints of the interval where threshold
                weights are positive. Must be increasing.
            weights: Array of weights to apply to the score (e.g., latitude
                weighting). If None, no weights are applied.
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        self.interval_where_one = interval_where_one
        self.interval_where_positive = interval_where_positive
        self.weights = weights
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Mean Absolute Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            The computed Mean Absolute Error result.
        """
        if self.interval_where_one is not None:
            return scores.continuous.tw_absolute_error(
                forecast,
                target,
                interval_where_one=self.interval_where_one,
                interval_where_positive=self.interval_where_positive,
                weights=self.weights,
                preserve_dims=self.preserve_dims,
            )
        return scores.continuous.mae(forecast, target, preserve_dims=self.preserve_dims)


class MeanError(BaseMetric):
    """Compute Mean Error (bias) between forecast and target.

    Extends BaseMetric to calculate mean error (bias) using the preserve_dims
    dimensions. Positive values indicate forecast exceeds target.
    """

    def __init__(self, name: str = "MeanError", *args, **kwargs):
        """Initialize the Mean Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanError".
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Mean Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            The computed Mean Error result.
        """
        return scores.continuous.mean_error(
            forecast, target, preserve_dims=self.preserve_dims
        )


class RootMeanSquaredError(BaseMetric):
    """Compute Root Mean Squared Error between forecast and target.

    Extends BaseMetric to calculate RMSE using the preserve_dims dimensions.
    RMSE is the square root of the mean squared error.
    """

    def __init__(self, name: str = "RootMeanSquaredError", *args, **kwargs):
        """Initialize the Root Mean Squared Error metric.

        Args:
            name: The name of the metric. Defaults to "RootMeanSquaredError".
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Root Mean Square Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            The computed Root Mean Square Error result.
        """
        return scores.continuous.rmse(
            forecast, target, preserve_dims=self.preserve_dims
        )


class EarlySignal(BaseMetric):
    """Detect first occurrence of signal exceeding threshold criteria.

    Extends BaseMetric to find the earliest time when a signal is detected
    based on threshold criteria, returning init_time, lead_time, and
    valid_time information. Flexible for different signal detection criteria.
    """

    def __init__(
        self,
        name: str = "EarlySignal",
        comparison_operator: Union[
            Callable, Literal[">", ">=", "<", "<=", "==", "!="]
        ] = ">=",
        threshold: float = 0.5,
        spatial_aggregation: Literal["any", "all", "half"] = "any",
        **kwargs,
    ):
        """Initialize the Early Signal detection metric.

        Args:
            name: The name of the metric. Defaults to "EarlySignal".
            comparison_operator: The comparison operator for signal detection.
            threshold: The threshold value for signal detection.
            spatial_aggregation: Spatial aggregation method. Options: "any"
                (any gridpoint meets criteria), "all" (all gridpoints meet
                criteria), or "half" (at least half meet criteria).
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        # Extract threshold params before passing to super
        self.comparison_operator = utils.maybe_get_operator(comparison_operator)
        self.threshold = threshold
        self.spatial_aggregation = spatial_aggregation
        super().__init__(name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataarray with init_time, lead_time, valid_time.
            target: The target dataarray (used for reference/validation).

        Returns:
            Boolean DataArray with dims [init_time, lead_time] indicating
            whether criteria are met for each init_time and lead_time pair.
        """
        if self.threshold is None:
            # Return False for all when no detection criteria specified
            dims = ["init_time", "lead_time"]
            coords = {
                "init_time": forecast.valid_time - forecast.lead_time,
                "lead_time": forecast.lead_time,
            }
            if "valid_time" in forecast.dims:
                dims.append("valid_time")
                coords["valid_time"] = forecast.valid_time
            return xr.DataArray(
                False,
                dims=dims,
                coords=coords,
                name=self.name,
            )
        # Create detection mask
        detection_mask = self.comparison_operator(forecast, self.threshold)

        # Apply spatial aggregation
        spatial_dims = [
            dim
            for dim in detection_mask.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]

        if spatial_dims:
            if self.spatial_aggregation == "any":
                detection_mask = detection_mask.any(spatial_dims)
            elif self.spatial_aggregation == "all":
                detection_mask = detection_mask.all(spatial_dims)
            elif self.spatial_aggregation == "half":
                detection_mask = operator.ge(detection_mask.mean(spatial_dims), 0.5)
            else:
                raise ValueError(
                    f"Spatial aggregation '{self.spatial_aggregation}' not supported"
                )

        detection_mask.name = self.name
        return detection_mask


class MaximumMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE between forecast and target maximum values.

    Extends MeanAbsoluteError to filter forecast to a time window around the
    target's maximum using tolerance_range_hours. Useful for evaluating peak
    value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        name: str = "MaximumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Maximum Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                maximum to search for forecast maximum. Defaults to 24.
            reduce_spatial_dims: Spatial dimensions to reduce. Defaults to
                ["latitude", "longitude"].
            name: The name of the metric. Defaults to
                "MaximumMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        self.tolerance_range_hours = tolerance_range_hours
        self.reduce_spatial_dims = reduce_spatial_dims
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute MaximumMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            MeanAbsoluteError of the maximum values.
        """
        # Enforced spatial reduction for MaximumMeanAbsoluteError
        reduce_spatial_dims = ["latitude", "longitude"]
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        ).compute()
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        filtered_max_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= maximum_timestep.data
                - np.timedelta64(self.tolerance_range_hours // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= maximum_timestep.data
                + np.timedelta64(self.tolerance_range_hours // 2, "h")
            ),
            drop=True,
        ).max("valid_time")
        return super()._compute_metric(
            forecast=filtered_max_forecast,
            target=maximum_value,
            preserve_dims=self.preserve_dims,
        )


class MinimumMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE between forecast and target minimum values.

    Extends MeanAbsoluteError to filter forecast to a time window around the
    target's minimum using tolerance_range_hours. Useful for evaluating
    minimum value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        name: str = "MinimumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Minimum Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                minimum to search for forecast minimum. Defaults to 24.
            reduce_spatial_dims: Spatial dimensions to reduce. Defaults to
                ["latitude", "longitude"].
            name: The name of the metric. Defaults to
                "MinimumMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        self.tolerance_range_hours = tolerance_range_hours
        self.reduce_spatial_dims = reduce_spatial_dims
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MinimumMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            MeanAbsoluteError of the minimum values.
        """
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
        filtered_min_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= minimum_timestep.data
                - np.timedelta64(self.tolerance_range_hours // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= minimum_timestep.data
                + np.timedelta64(self.tolerance_range_hours // 2, "h")
            ),
            drop=True,
        ).min("valid_time")
        return super()._compute_metric(
            forecast=filtered_min_forecast,
            target=minimum_value,
            preserve_dims=self.preserve_dims,
        )


class MaximumLowestMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE of maximum aggregated minimum values for heatwaves.

    Extends MeanAbsoluteError for heatwave evaluation by aggregating daily
    minimum values and computing MAE between the warmest nighttime (daily
    minimum) temperature in target and forecast.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        name: str = "MaximumLowestMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Maximum Lowest Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                max-min value to search for forecast max-min. Defaults to 24.
            name: The name of the metric. Defaults to
                "MaximumLowestMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        self.tolerance_range_hours = tolerance_range_hours
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MaximumLowestMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            MeanAbsoluteError of the highest aggregated minimum value.
        """
        reduce_dims = [
            dim
            for dim in forecast.dims
            if dim not in ["valid_time", "lead_time", "time"]
        ]
        forecast = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=reduce_dims, skipna=True
        )
        target = utils.reduce_dataarray(
            target, method="mean", reduce_dims=reduce_dims, skipna=True
        )

        time_resolution_hours = utils.determine_temporal_resolution(target)
        max_min_target_value = (
            target.groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present,
                time_resolution_hours=time_resolution_hours,
            )
            .max()
        )
        max_min_target_datetime = target.where(
            target == max_min_target_value, drop=True
        ).valid_time

        # Handle the case where there are >1 resulting target values
        max_min_target_datetime = (
            utils.maybe_get_closest_timestamp_to_center_of_valid_times(
                max_min_target_datetime, target.valid_time
            )
        )
        subset_forecast = (
            forecast.where(
                (
                    forecast.valid_time
                    >= (
                        max_min_target_datetime.data
                        - np.timedelta64(self.tolerance_range_hours // 2, "h")
                    )
                )
                & (
                    forecast.valid_time
                    <= (
                        max_min_target_datetime.data
                        + np.timedelta64(self.tolerance_range_hours // 2, "h")
                    )
                ),
                drop=True,
            )
            .groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present_forecast,
                time_resolution_hours=utils.determine_temporal_resolution(forecast),
            )
            .min("dayofyear")
        )

        return super()._compute_metric(
            forecast=subset_forecast,
            target=max_min_target_value,
            preserve_dims=self.preserve_dims,
        )


class DurationMeanError(MeanError):
    """Compute mean error of event duration between forecast and target.

    Extends MeanError to compute the mean error between forecast and target
    event durations based on threshold criteria and spatial aggregation.
    """

    def __init__(
        self,
        threshold_criteria: xr.DataArray | float,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        op_func: Union[Callable, Literal[">", ">=", "<", "<=", "==", "!="]] = ">=",
        name: str = "DurationMeanError",
        preserve_dims: str = "init_time",
        product_time_resolution_hours: bool = False,
    ):
        """Initialize the Duration Mean Error metric.

        Args:
            threshold_criteria: Criteria for event detection. Either a
                DataArray of climatology with dimensions (dayofyear, hour,
                latitude, longitude) or a float fixed threshold.
            reduce_spatial_dims: Spatial dimensions to reduce prior to
                applying threshold criteria. Defaults to ["latitude",
                "longitude"].
            op_func: Comparison operator or string (e.g., operator.ge for
                >=).
            name: Name of the metric. Defaults to "DurationMeanError".
            preserve_dims: Dimensions to preserve during aggregation.
                Defaults to "init_time".
            product_time_resolution_hours: Whether to multiply duration by
                time resolution of forecast (in hours). Defaults to False.
        """
        super().__init__(name=name, preserve_dims=preserve_dims)
        self.reduce_spatial_dims = reduce_spatial_dims
        self.threshold_criteria = threshold_criteria
        self.op_func = utils.maybe_get_operator(op_func)
        self.product_time_resolution_hours = product_time_resolution_hours

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> Any:
        """Compute spatially averaged duration mean error.

        Args:
            forecast: the forecast DataArray.
            target: the target DataArray.

        Returns:
            The mean error between forecast and target event durations.
        """
        # Handle criteria - either climatology (xr.DataArray) or float threshold
        # Use local variable to avoid mutating self.threshold_criteria
        threshold_criteria = self.threshold_criteria

        # Need to get climatology into the correct format and interpolation for
        # comparison
        if isinstance(threshold_criteria, xr.DataArray):
            # Climatology case, convert from dayofyear/hour to valid_time.
            # Note that unintended behavior may occur if the case spans multiple years.
            threshold_criteria = utils.convert_day_yearofday_to_time(
                threshold_criteria, forecast.valid_time.dt.year.values[0]
            )

            # Interpolate climatology to target coordinates
            threshold_criteria = utils.interp_climatology_to_target(
                target, threshold_criteria
            )
        # Reduce spatial dimensions if specified (default is to reduce)
        if len(self.reduce_spatial_dims) > 0:
            target = utils.reduce_dataarray(
                target, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
            )
            forecast = utils.reduce_dataarray(
                forecast,
                method="mean",
                reduce_dims=self.reduce_spatial_dims,
                skipna=True,
            )

            if isinstance(threshold_criteria, xr.DataArray):
                threshold_criteria = utils.reduce_dataarray(
                    threshold_criteria,
                    method="mean",
                    reduce_dims=self.reduce_spatial_dims,
                    skipna=True,
                )
        forecast_mask = self.op_func(forecast, threshold_criteria)
        target_mask = self.op_func(target, threshold_criteria)

        # Track NaN locations in forecast data
        forecast_valid_mask = ~forecast.isnull()

        # Apply valid data mask (exclude NaN positions in forecast)
        forecast_mask_final = forecast_mask.where(forecast_valid_mask)
        try:
            target_mask_final = target_mask.where(forecast_valid_mask)
        # If sparse, will need to expand_dims first as transpose is not supported
        except AttributeError:
            logger.info(
                "Target mask is sparse, expanding dimensions to handle unsupported "
                "transpose operation."
            )
            target_mask_final = target_mask.expand_dims(
                dim={"lead_time": target.lead_time.size}
            ).where(forecast_valid_mask)

        # Sum to get durations (NaN values are excluded by default)
        forecast_duration = forecast_mask_final.groupby(self.preserve_dims).sum(
            skipna=True
        )
        target_duration = target_mask_final.groupby(self.preserve_dims).sum(skipna=True)

        if self.product_time_resolution_hours:
            time_resolution_hours = utils.determine_temporal_resolution(forecast)
            forecast_duration = forecast_duration * time_resolution_hours
            target_duration = target_duration * time_resolution_hours

        return super()._compute_metric(
            forecast=forecast_duration,
            target=target_duration,
            preserve_dims=self.preserve_dims,
        )


_CLIMO_DS = None  # lazy handle (no data loaded)


def _open_t2m_climatology() -> xr.Dataset:
    """Return a lazily-opened handle to the climatology file.

    The file is opened once and reused; no data is read until a
    ``.sel()`` / ``.load()`` call materialises a subset.
    """
    global _CLIMO_DS
    if _CLIMO_DS is not None:
        return _CLIMO_DS
    if not CLIMO_T2M_PATH.exists():
        raise FileNotFoundError(f"Climatology file not found: {CLIMO_T2M_PATH}")
    _CLIMO_DS = xr.open_dataset(CLIMO_T2M_PATH)
    return _CLIMO_DS


def _load_t2m_climatology() -> xr.Dataset:
    """Legacy helper — returns the lazily-opened climatology.

    Callers that need an in-memory subset should use
    ``_open_t2m_climatology()`` and subset/load explicitly.
    """
    return _open_t2m_climatology()


def _extract_first_event_window(
    mask_1d: np.ndarray,
    min_steps: int,
) -> tuple[int, int] | None:
    """Return (start_idx, end_idx_exclusive) of first run meeting min_steps."""
    if mask_1d.size == 0:
        return None
    start = None
    run = 0
    for i, val in enumerate(mask_1d.astype(bool)):
        if val:
            if start is None:
                start = i
            run += 1
            if run >= min_steps:
                j = i + 1
                while j < mask_1d.size and bool(mask_1d[j]):
                    j += 1
                return int(start), int(j)
        else:
            start = None
            run = 0
    return None


def _find_events_vectorized(
    mask: np.ndarray,
    min_steps: int,
    time_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized first-qualifying-event detection along *time_axis*.

    Parameters
    ----------
    mask : bool ndarray, shape ``(..., n_time, ...)``
        Boolean exceedance mask (True where threshold is exceeded).
    min_steps : int
        Minimum contiguous True run length that qualifies as an event.
    time_axis : int
        Axis corresponding to the time dimension (default 0).

    Returns
    -------
    has_event : bool ndarray
        Same shape as *mask* minus the *time_axis* dimension.
        True where a qualifying event was found.
    event_start : int ndarray (same shape as has_event)
        Index of the first step of the qualifying event (meaningless where
        *has_event* is False).
    event_end : int ndarray (same shape as has_event)
        Exclusive end index of the qualifying contiguous run (meaningless
        where *has_event* is False).
    """
    # Move time axis to position 0 for uniform indexing.
    m = np.moveaxis(mask.astype(np.int32), time_axis, 0)
    n_time = m.shape[0]
    spatial_shape = m.shape[1:]

    # --- Step 1: rolling sum via cumsum to find first qualifying window ------
    cs = np.cumsum(m, axis=0)                        # (n_time, ...)
    rolling = cs.copy()
    if min_steps < n_time:
        rolling[min_steps:] = cs[min_steps:] - cs[:-min_steps]
    # rolling[i] == min_steps iff mask[i-min_steps+1 .. i] are all True.

    qualifying = (rolling == min_steps)               # bool (n_time, ...)
    has_event = qualifying.any(axis=0)                # (...)
    # argmax returns index of first True; 0 when all-False (masked out below).
    first_qualify_idx = qualifying.argmax(axis=0)     # (...)

    # The contiguous block genuinely starts at first_qualify_idx - min_steps + 1
    # because if position (idx-1) also had rolling==min_steps, argmax would have
    # picked it instead.
    event_start = first_qualify_idx - min_steps + 1   # (...)

    # --- Step 2: find the full extent (first False after first_qualify_idx) ---
    time_idx = np.arange(n_time).reshape(
        (n_time,) + (1,) * len(spatial_shape)
    )                                                 # (n_time, 1, 1, ...)
    # Broadcast first_qualify_idx for comparison.
    fq = first_qualify_idx[np.newaxis, ...]           # (1, ...)
    after_qualify = time_idx > fq                     # (n_time, ...)
    false_after = (~m.astype(bool)) & after_qualify   # (n_time, ...)

    has_false = false_after.any(axis=0)               # (...)
    # Where false_after is True, keep the time index; elsewhere use n_time as sentinel.
    first_false_idx = np.where(false_after, time_idx, n_time).min(axis=0)
    event_end = np.where(has_false, first_false_idx, n_time)  # exclusive

    return has_event, event_start, event_end


def _find_end_from_start(
    mask: np.ndarray,
    time_axis: int = 0,
) -> np.ndarray:
    """Find the first False along *time_axis* (event end for inherited events).

    Used by rolling event inheritance: when we know the event is already active
    at step 0, the "end" is simply the first time step that drops below
    threshold.

    Returns
    -------
    end_idx : int ndarray, shape = mask.shape minus *time_axis*
        Exclusive end index.  If the mask is True for the entire time range,
        returns ``n_time`` (event runs to the end of the forecast window).
    """
    m = np.moveaxis(mask.astype(bool), time_axis, 0)
    n_time = m.shape[0]
    spatial_shape = m.shape[1:]
    false_mask = ~m
    has_false = false_mask.any(axis=0)
    time_idx = np.arange(n_time).reshape(
        (n_time,) + (1,) * len(spatial_shape)
    )
    first_false = np.where(false_mask, time_idx, n_time).min(axis=0)
    return np.where(has_false, first_false, n_time)


def _to_daily_agg(
    *arrays: np.ndarray,
    valid_times: np.ndarray,
    agg: str = "max",
) -> list[np.ndarray]:
    """Aggregate sub-daily arrays to daily max or min by calendar day.

    Parameters
    ----------
    *arrays : ndarray, shape ``(n_time, ...spatial)``
        One or more arrays to aggregate along axis 0.
    valid_times : ndarray of datetime64
        Valid time for each position along the time axis (length ``n_time``).
    agg : {"max", "min"}
        Aggregation function — ``"max"`` for heat waves, ``"min"`` for freezes.

    Returns
    -------
    daily_arrays : list of ndarray, shape ``(n_days, ...spatial)``
        One array per input, aggregated to daily max/min.
    """
    agg_fn = np.nanmax if agg == "max" else np.nanmin
    days = np.asarray(valid_times, dtype="datetime64[D]")
    unique_days, day_idx = np.unique(days, return_inverse=True)
    n_days = len(unique_days)

    results: list[np.ndarray] = []
    for arr in arrays:
        spatial_shape = arr.shape[1:]
        daily = np.full((n_days, *spatial_shape), np.nan, dtype=arr.dtype)
        for di in range(n_days):
            mask = day_idx == di
            daily[di] = agg_fn(arr[mask], axis=0)
        results.append(daily)
    return results


def _to_daily_max(
    *arrays: np.ndarray,
    valid_times: np.ndarray,
) -> list[np.ndarray]:
    """Aggregate sub-daily arrays to daily maxima by calendar day."""
    return _to_daily_agg(*arrays, valid_times=valid_times, agg="max")


def _to_daily_min(
    *arrays: np.ndarray,
    valid_times: np.ndarray,
) -> list[np.ndarray]:
    """Aggregate sub-daily arrays to daily minima by calendar day."""
    return _to_daily_agg(*arrays, valid_times=valid_times, agg="min")


class ClimatologyEventTimingError(BaseMetric):
    """Onset/end/duration error for heat/freeze events using climatology thresholds.

    Heat-wave threshold (Perkins & Alexander 2013, definition 1):
      daily Tmax >= P90 of daily-Tmax climatology for >= 3 consecutive days,
      where P90 = climatology_mean + 1.282 * climatology_std (normal approx).
      Sub-daily data is aggregated to daily maxima before event detection.
    Freeze threshold:
      daily Tmin <= P5 of climatology for >= 3 consecutive days,
      where P5 = climatology_mean - 1.645 * climatology_std (normal approx).
      Sub-daily data is aggregated to daily minima before event detection.
    """

    def __init__(
        self,
        event_kind: Literal["heat_wave", "freeze"],
        metric_component: Literal["onset", "end", "duration", "peak_timing_bias", "peak_timing_rmse"],
        name: str | None = None,
        preserve_dims: str = "init_time",
        reduce_spatial_dims: list[str] | None = None,
        **kwargs,
    ):
        if name is None:
            suffix = {
                "onset": "onset_error",
                "end": "end_error",
                "duration": "duration_error",
                "peak_timing_bias": "peak_timing_bias",
                "peak_timing_rmse": "peak_timing_rmse",
            }[metric_component]
            name = f"{event_kind}_{suffix}"
        if reduce_spatial_dims is None:
            reduce_spatial_dims = ["latitude", "longitude"]
        super().__init__(name=name, preserve_dims=preserve_dims, **kwargs)
        self.event_kind = event_kind
        self.metric_component = metric_component
        self.reduce_spatial_dims = reduce_spatial_dims

    def _threshold_from_climo(
        self,
        data: xr.DataArray,
        valid_time: xr.DataArray,
    ) -> xr.DataArray:
        import time as _t
        _t0 = _t.perf_counter()

        climo_full = _open_t2m_climatology()

        # ---- 1. Determine spatial subset ------------------------------------
        # Build lat/lon slices to avoid loading the full 0.25° global grid.
        pad = 1.0  # degree padding around forecast extent
        lat_name = lon_name = None
        for ln, lo in [("latitude", "longitude"), ("lat", "lon")]:
            if ln in data.dims:
                lat_name, lon_name = ln, lo
                break
        stn_lats = stn_lons = None
        if lat_name is None and "station" in data.dims:
            if "station_latitude" in data.coords and "station_longitude" in data.coords:
                stn_lats = data["station_latitude"].values
                stn_lons = data["station_longitude"].values

        spatial_sel: dict = {}
        if lat_name and lat_name in climo_full.dims:
            lat_vals = data[lat_name].values
            lon_vals = data[lon_name].values
            climo_lat = climo_full[lat_name].values
            # Handle descending latitude
            lat_lo, lat_hi = float(lat_vals.min()) - pad, float(lat_vals.max()) + pad
            if climo_lat[0] > climo_lat[-1]:
                spatial_sel[lat_name] = slice(lat_hi, lat_lo)
            else:
                spatial_sel[lat_name] = slice(lat_lo, lat_hi)
            lon_lo, lon_hi = float(lon_vals.min()) - pad, float(lon_vals.max()) + pad
            spatial_sel[lon_name] = slice(lon_lo, lon_hi)
        elif stn_lats is not None and "latitude" in climo_full.dims:
            lat_lo = float(stn_lats.min()) - pad
            lat_hi = float(stn_lats.max()) + pad
            lon_lo = float(stn_lons.min()) - pad
            lon_hi = float(stn_lons.max()) + pad
            climo_lat = climo_full["latitude"].values
            if climo_lat[0] > climo_lat[-1]:
                spatial_sel["latitude"] = slice(lat_hi, lat_lo)
            else:
                spatial_sel["latitude"] = slice(lat_lo, lat_hi)
            spatial_sel["longitude"] = slice(lon_lo, lon_hi)

        # ---- 2. Determine DOY subset ----------------------------------------
        doy_mmdd = valid_time.dt.month * 100 + valid_time.dt.day
        hour = valid_time.dt.hour
        unique_doy = np.unique(np.asarray(doy_mmdd.values).ravel())
        unique_hr = np.unique(np.asarray(hour.values).ravel())

        # ---- 3. Subset and load only what we need ---------------------------
        climo_sub = climo_full
        if spatial_sel:
            climo_sub = climo_sub.sel(**spatial_sel)
        climo_sub = climo_sub.sel(DOY=unique_doy, hour=unique_hr, method="nearest")
        climo_sub = climo_sub.load()  # materialise the small subset

        mean = climo_sub["t2m_mean"]
        std = climo_sub["t2m_std"]

        # ---- 4. Select per valid-time (now fast, in-memory) ------------------
        climo_selected = mean.sel(DOY=doy_mmdd, hour=hour, method="nearest")
        std_selected = std.sel(DOY=doy_mmdd, hour=hour, method="nearest")
        if self.event_kind == "heat_wave":
            threshold = climo_selected + 1.282 * std_selected
        else:
            threshold = climo_selected - 1.645 * std_selected

        _elapsed = _t.perf_counter() - _t0
        print(
            f"[METRIC:{self.name}] threshold built in {_elapsed:.2f}s "
            f"(climo subset: DOY={len(unique_doy)}, hr={len(unique_hr)}, "
            f"spatial={dict(climo_sub.sizes)})",
            flush=True,
        )

        # ---- 5. Interpolate to forecast grid / stations ----------------------
        if (
            lat_name
            and lat_name in data.dims
            and lat_name in threshold.dims
        ):
            return threshold.interp(
                {lat_name: data[lat_name], lon_name: data[lon_name]},
                method="nearest",
                kwargs={"fill_value": None},
            )

        if (
            "station" in data.dims
            and "station_latitude" in data.coords
            and "station_longitude" in data.coords
            and "latitude" in threshold.dims
            and "longitude" in threshold.dims
        ):
            return threshold.interp(
                latitude=xr.DataArray(stn_lats, dims="station"),
                longitude=xr.DataArray(stn_lons, dims="station"),
                method="nearest",
                kwargs={"fill_value": None},
            )

        reduce_spatial = [
            d for d in ("latitude", "longitude", "lat", "lon")
            if d in threshold.dims
        ]
        if reduce_spatial:
            threshold = threshold.mean(dim=reduce_spatial, skipna=True)
        return threshold

    def _component_error_for_series(
        self,
        forecast_values: np.ndarray,
        target_values: np.ndarray,
        threshold_values: np.ndarray,
        time_resolution_hours: float,
    ) -> float:
        onset_err, end_err, duration_err = self._all_component_errors_for_series(
            forecast_values=forecast_values,
            target_values=target_values,
            threshold_values=threshold_values,
            time_resolution_hours=time_resolution_hours,
        )
        if self.metric_component == "onset":
            return onset_err
        if self.metric_component == "end":
            return end_err
        return duration_err

    def _all_component_errors_for_series(
        self,
        forecast_values: np.ndarray,
        target_values: np.ndarray,
        threshold_values: np.ndarray,
        time_resolution_hours: float,
    ) -> tuple[float, float, float]:
        # NOTE: This fallback path does not yet perform daily-max aggregation
        # for heat waves.  The main vectorized path (compute_all_components)
        # handles this correctly; this path is only reached for rare
        # single-series evaluations.
        if self.event_kind == "heat_wave":
            fmask = forecast_values >= threshold_values
            tmask = target_values >= threshold_values
            min_days = 3
        else:
            fmask = forecast_values <= threshold_values
            tmask = target_values <= threshold_values
            min_days = 3

        min_steps = max(1, int(np.ceil(min_days * 24.0 / float(time_resolution_hours))))
        fwin = _extract_first_event_window(fmask, min_steps=min_steps)
        twin = _extract_first_event_window(tmask, min_steps=min_steps)
        if fwin is None or twin is None:
            nan = float("nan")
            return nan, nan, nan

        f_start, f_end = fwin
        t_start, t_end = twin
        onset = float((f_start - t_start) * time_resolution_hours)
        # Compare final inclusive event step timing.
        end = float(((f_end - 1) - (t_end - 1)) * time_resolution_hours)
        duration = float(((f_end - f_start) - (t_end - t_start)) * time_resolution_hours)
        return onset, end, duration

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        # If a composite precomputed all components, reuse them directly.
        precomputed = kwargs.get("__event_timing_components")
        if isinstance(precomputed, dict):
            maybe_da = precomputed.get(self.metric_component)
            if isinstance(maybe_da, xr.DataArray):
                return maybe_da

        components = self.compute_all_components(
            forecast,
            target,
            debug_values=bool(kwargs.get("debug_heat_values", False)),
            debug_max_inits=int(kwargs.get("debug_heat_max_inits", 2)),
            debug_max_points=int(kwargs.get("debug_heat_max_points", 3)),
        )
        return components[self.metric_component]

    def compute_all_components(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        debug_values: bool = False,
        debug_max_inits: int = 2,
        debug_max_points: int = 3,
    ) -> dict[str, xr.DataArray]:
        """Compute onset/end/duration errors together in one vectorized pass.

        Uses array-wide rolling-sum event detection instead of Python loops
        over spatial points.  The only remaining loop is over ``init_time``
        (typically ~30-40 iterations), and within each iteration all spatial
        points are processed simultaneously via numpy bulk ops.
        """
        import time as _time

        t0 = _time.perf_counter()

        if debug_values:
            print(
                f"[METRIC:{self.name}] input sizes forecast={dict(forecast.sizes)} "
                f"target={dict(target.sizes)}",
                flush=True,
            )

        # ---- optional spatial reduction (empty list → grid-first) -----------
        reduce_dims = [
            d
            for d in self.reduce_spatial_dims
            if d in forecast.dims and d in target.dims
        ]
        if self.reduce_spatial_dims and not reduce_dims:
            time_like_dims = {"init_time", "lead_time", "valid_time", "time", "dayofyear", "hour"}
            reduce_dims = [
                d for d in forecast.dims if d in target.dims and d not in time_like_dims
            ]

        if reduce_dims:
            forecast_r = utils.reduce_dataarray(forecast, method="mean", reduce_dims=reduce_dims, skipna=True)
            target_r = utils.reduce_dataarray(target, method="mean", reduce_dims=reduce_dims, skipna=True)
        else:
            forecast_r = forecast
            target_r = target

        # ---- valid_time & threshold -----------------------------------------
        if "valid_time" in forecast_r.coords:
            valid_time = forecast_r["valid_time"]
        elif "init_time" in forecast_r.coords and "lead_time" in forecast_r.coords:
            valid_time = forecast_r["init_time"] + forecast_r["lead_time"]
        else:
            raise ValueError(
                f"{self.name}: forecast requires valid_time coordinate or init_time+lead_time."
            )

        threshold = self._threshold_from_climo(forecast_r, valid_time)
        time_resolution_hours = utils.determine_temporal_resolution(forecast_r)

        if debug_values:
            print(
                f"[METRIC:{self.name}] after threshold: forecast={dict(forecast_r.sizes)} "
                f"target={dict(target_r.sizes)} threshold={dict(threshold.sizes)}",
                flush=True,
            )

        # ---- reconstruct init_time if needed --------------------------------
        if (
            "init_time" not in forecast_r.dims
            and {"lead_time", "valid_time"}.issubset(set(forecast_r.dims))
        ):
            forecast_r = utils.convert_valid_time_to_init_time(forecast_r)
            if "lead_time" not in target_r.dims and "lead_time" in forecast_r.dims:
                target_r = target_r.expand_dims(lead_time=forecast_r["lead_time"])
            target_r = utils.convert_valid_time_to_init_time(target_r)
            if "lead_time" not in threshold.dims and "lead_time" in forecast_r.dims:
                threshold = threshold.expand_dims(lead_time=forecast_r["lead_time"])
            threshold = utils.convert_valid_time_to_init_time(threshold)

            forecast_r, target_r, threshold = xr.align(
                forecast_r, target_r, threshold, join="inner"
            )
            if debug_values:
                print(
                    f"[METRIC:{self.name}] reconstructed init_time: "
                    f"forecast={dict(forecast_r.sizes)} target={dict(target_r.sizes)} "
                    f"threshold={dict(threshold.sizes)}",
                    flush=True,
                )

        # ---- event parameters -----------------------------------------------
        # For heat waves we aggregate sub-daily data to daily Tmax before event
        # detection (Perkins & Alexander 2013: daily Tmax >= P90 for >=3 days).
        # For freezes, aggregate to daily Tmin (daily Tmin <= P5 for >=3 days).
        _raw_time_res_hours = time_resolution_hours  # original sub-daily resolution
        _use_daily_agg = self.event_kind in ("heat_wave", "freeze")
        _daily_agg_kind = "max" if self.event_kind == "heat_wave" else "min"
        if _use_daily_agg:
            min_days = 3
            time_resolution_hours = 24.0   # event detection on daily data
            min_steps = 3
        else:
            min_days = 3
            min_steps = max(1, int(np.ceil(min_days * 24.0 / float(time_resolution_hours))))

        # ---- helper: vectorized event errors for a single init slice --------
        def _vectorized_errors_for_slice(
            f_arr: np.ndarray,
            t_arr: np.ndarray,
            thr_arr: np.ndarray,
        ) -> dict:
            """Return event errors + detection state for a single init slice.

            All inputs have shape ``(n_time, ...spatial...)`` with time on axis 0.
            Event detection, timing errors, and peak-timing errors are fully
            vectorized across spatial dimensions.

            Returns a dict with keys:
                onset, end, duration, peak_bias, peak_rmse  – float errors
                f_mask, t_mask  – bool exceedance masks (n_time, ...spatial)
                f_has, t_has    – bool event-found flags (...spatial)
                f_end, t_end    – int end indices (...spatial)
            """
            n_time = f_arr.shape[0]
            spatial_shape = f_arr.shape[1:]
            NAN_ERRORS = dict(
                onset=np.nan, end=np.nan, duration=np.nan,
                peak_bias=np.nan, peak_rmse=np.nan,
            )

            # Boolean exceedance masks
            valid = np.isfinite(f_arr) & np.isfinite(t_arr) & np.isfinite(thr_arr)
            if self.event_kind == "heat_wave":
                f_mask = (f_arr >= thr_arr) & valid
                t_mask = (t_arr >= thr_arr) & valid
            else:
                f_mask = (f_arr <= thr_arr) & valid
                t_mask = (t_arr <= thr_arr) & valid

            # Vectorized event detection along time axis (axis=0)
            f_has, f_start, f_end = _find_events_vectorized(f_mask, min_steps, time_axis=0)
            t_has, t_start, t_end = _find_events_vectorized(t_mask, min_steps, time_axis=0)

            # Always include detection state for inheritance
            state = dict(
                f_mask=f_mask, t_mask=t_mask,
                f_has=f_has, t_has=t_has,
                f_start=f_start, t_start=t_start,
                f_end=f_end, t_end=t_end,
            )

            both = f_has & t_has  # (...spatial...)
            missed = t_has & (~f_has)  # obs detected, fc missed
            n_both = int(both.sum())
            n_missed = int(missed.sum())
            n_target = int(t_has.sum())
            n_contributing = n_both + n_missed

            if n_contributing == 0:
                return {**NAN_ERRORS, "n_event_points": 0,
                        "n_target_events": n_target, **state}

            # Observed event duration at missed points (for penalty)
            obs_dur_missed = (
                (t_end[missed].astype(float) - t_start[missed].astype(float))
                * time_resolution_hours
            ) if n_missed > 0 else np.array([])

            # --- Normal errors for 'both' points ---------------------------------
            if n_both > 0:
                f_s = f_start[both].astype(float)
                t_s = t_start[both].astype(float)
                f_e = f_end[both].astype(float)
                t_e = t_end[both].astype(float)
                onset_det = (f_s - t_s) * time_resolution_hours
                end_det = (f_e - t_e) * time_resolution_hours
                dur_det = ((f_e - f_s) - (t_e - t_s)) * time_resolution_hours

                time_idx = np.arange(n_time).reshape(
                    (n_time,) + (1,) * len(spatial_shape)
                )
                f_in_event = (
                    (time_idx >= f_start[np.newaxis, ...])
                    & (time_idx < f_end[np.newaxis, ...])
                    & f_has[np.newaxis, ...]
                )
                if self.event_kind == "heat_wave":
                    f_for_peak = np.where(f_in_event, f_arr, -np.inf)
                    f_peak_idx = f_for_peak.argmax(axis=0)
                else:
                    f_for_peak = np.where(f_in_event, f_arr, np.inf)
                    f_peak_idx = f_for_peak.argmin(axis=0)
                t_in_event = (
                    (time_idx >= t_start[np.newaxis, ...])
                    & (time_idx < t_end[np.newaxis, ...])
                    & t_has[np.newaxis, ...]
                )
                if self.event_kind == "heat_wave":
                    t_for_peak = np.where(t_in_event, t_arr, -np.inf)
                    t_peak_idx = t_for_peak.argmax(axis=0)
                else:
                    t_for_peak = np.where(t_in_event, t_arr, np.inf)
                    t_peak_idx = t_for_peak.argmin(axis=0)
                peak_det = (f_peak_idx[both].astype(float) - t_peak_idx[both].astype(float)) * time_resolution_hours
            else:
                onset_det = np.array([])
                end_det = np.array([])
                dur_det = np.array([])
                peak_det = np.array([])

            # --- Penalty errors for missed points --------------------------------
            onset_pen = obs_dur_missed          # late by entire event
            end_pen = -obs_dur_missed           # ended early by entire event
            dur_pen = -obs_dur_missed           # predicted 0 of N hours
            peak_pen = obs_dur_missed           # peak timing off by entire event

            # --- Combined means ---------------------------------------------------
            all_onset = np.concatenate([onset_det, onset_pen])
            all_end = np.concatenate([end_det, end_pen])
            all_dur = np.concatenate([dur_det, dur_pen])
            all_peak = np.concatenate([peak_det, peak_pen])

            return {
                "onset": float(np.nanmean(all_onset)),
                "end": float(np.nanmean(all_end)),
                "duration": float(np.nanmean(all_dur)),
                "peak_bias": float(np.nanmean(all_peak)),
                "peak_rmse": float(np.sqrt(np.nanmean(all_peak ** 2))),
                "n_event_points": n_both,
                "n_target_events": n_target,
                **state,
            }

        # =====================================================================
        # MAIN PATH: init_time present → one vectorized call per init
        # =====================================================================
        if "init_time" in forecast_r.dims:
            forecast_r, target_r, threshold = xr.align(
                forecast_r, target_r, threshold, join="inner"
            )
            # Identify the time axis used for event detection within each init
            # After init_time reconstruction this is typically lead_time.
            time_dim = None
            for candidate in ("lead_time", "valid_time"):
                if candidate in forecast_r.dims:
                    time_dim = candidate
                    break
            if time_dim is None:
                # Degenerate: only init_time, no time series to scan.
                nan_da = xr.DataArray(
                    np.full(forecast_r.sizes["init_time"], np.nan),
                    dims=("init_time",),
                    coords={"init_time": forecast_r["init_time"]},
                )
                return {
                    "onset": nan_da.copy(), "end": nan_da.copy(), "duration": nan_da.copy(),
                    "peak_timing_bias": nan_da.copy(), "peak_timing_rmse": nan_da.copy(),
                }

            # Determine axis ordering: we want (init, time, ...spatial)
            # Transpose so that init_time is axis 0, time_dim is axis 1.
            dim_order = ["init_time", time_dim] + [
                d for d in forecast_r.dims if d not in ("init_time", time_dim)
            ]
            f_np = np.asarray(forecast_r.transpose(*dim_order).values, dtype=float)
            t_np = np.asarray(
                target_r.broadcast_like(forecast_r).transpose(*dim_order).values, dtype=float
            )
            thr_np = np.asarray(
                threshold.broadcast_like(forecast_r).transpose(*dim_order).values, dtype=float
            )
            # f_np shape: (n_init, n_time, ...spatial...)
            n_init = f_np.shape[0]
            spatial_shape = f_np.shape[2:]  # everything after (init, time)

            onset_errors = np.full(n_init, np.nan)
            end_errors = np.full(n_init, np.nan)
            duration_errors = np.full(n_init, np.nan)
            peak_bias_arr = np.full(n_init, np.nan)
            peak_rmse_arr = np.full(n_init, np.nan)
            inherited_flag = np.zeros(n_init, dtype=bool)
            n_event_pts = np.zeros(n_init, dtype=int)
            n_target_evts = np.zeros(n_init, dtype=int)

            print(
                f"[METRIC:{self.name}] evaluating {n_init} init_times "
                f"(vectorized, min_steps={min_steps}) for event_kind={self.event_kind}",
                flush=True,
            )

            # --- Rolling event inheritance ------------------------------------
            # Process init times chronologically so that each init can inherit
            # event status from the previous one.  This lets us compute
            # end-timing errors for inits that are too late to detect a full
            # event on their own.
            init_times_np = np.asarray(forecast_r["init_time"].values)
            sorted_order = np.argsort(init_times_np)  # chronological

            # Lead-time values needed for daily aggregation.
            if _use_daily_agg and time_dim == "lead_time":
                _lead_time_vals = forecast_r["lead_time"].values
            else:
                _lead_time_vals = None

            # Per-spatial-point "event active" state carried across inits.
            f_active = np.zeros(spatial_shape, dtype=bool)
            t_active = np.zeros(spatial_shape, dtype=bool)

            for idx in sorted_order:
                # --- Daily aggregation (max for heat waves, min for freezes) --
                if _use_daily_agg:
                    if _lead_time_vals is not None:
                        vt = init_times_np[idx] + _lead_time_vals
                    else:
                        vt = np.asarray(forecast_r["valid_time"].values)
                    f_day, t_day, thr_day = _to_daily_agg(
                        f_np[idx], t_np[idx], thr_np[idx],
                        valid_times=vt,
                        agg=_daily_agg_kind,
                    )
                    res = _vectorized_errors_for_slice(f_day, t_day, thr_day)
                else:
                    res = _vectorized_errors_for_slice(f_np[idx], t_np[idx], thr_np[idx])
                onset_errors[idx] = res["onset"]
                end_errors[idx] = res["end"]
                duration_errors[idx] = res["duration"]
                peak_bias_arr[idx] = res["peak_bias"]
                peak_rmse_arr[idx] = res["peak_rmse"]
                n_event_pts[idx] = res["n_event_points"]
                n_target_evts[idx] = res["n_target_events"]

                f_has = res["f_has"]   # (...spatial) bool
                t_has = res["t_has"]
                f_mask = res["f_mask"]  # (n_time, ...spatial) bool
                t_mask = res["t_mask"]

                # --- Inheritance check for end timing -------------------------
                # For spatial points that did NOT detect a full event, check
                # whether the event can be inherited from the previous init.
                f_first_exceeds = f_mask[0]  # (...spatial)
                t_first_exceeds = t_mask[0]
                f_can_inherit = f_active & (~f_has) & f_first_exceeds
                t_can_inherit = t_active & (~t_has) & t_first_exceeds

                # Points eligible for inherited end-timing: BOTH sides must
                # have an event (detected OR inherited).
                f_has_or_inh = f_has | f_can_inherit
                t_has_or_inh = t_has | t_can_inherit
                inherit_only = f_can_inherit | t_can_inherit  # at least one side inherited
                both_any = f_has_or_inh & t_has_or_inh & inherit_only

                if both_any.any():
                    # Some points can inherit event status.  Recompute end
                    # error by combining: detected + inherited + penalty.
                    f_end_inh = _find_end_from_start(f_mask, time_axis=0)
                    t_end_inh = _find_end_from_start(t_mask, time_axis=0)
                    end_err_inh = (
                        f_end_inh[both_any].astype(float)
                        - t_end_inh[both_any].astype(float)
                    ) * time_resolution_hours

                    parts = [end_err_inh]

                    # Directly detected points (not part of inheritance set)
                    both_det = f_has & t_has
                    detected_not_inherited = both_det & (~both_any)
                    if detected_not_inherited.any():
                        f_end_det = res["f_end"]
                        t_end_det = res["t_end"]
                        f_start_det = res["f_start"]
                        t_start_det = res["t_start"]
                        end_det = (
                            f_end_det[detected_not_inherited].astype(float)
                            - t_end_det[detected_not_inherited].astype(float)
                        ) * time_resolution_hours
                        parts.append(end_det)

                    # Points still missed after inheritance
                    still_missed = t_has & (~f_has_or_inh)
                    if still_missed.any():
                        t_start_res = res["t_start"]
                        t_end_res = res["t_end"]
                        obs_dur_still = (
                            t_end_res[still_missed].astype(float)
                            - t_start_res[still_missed].astype(float)
                        ) * time_resolution_hours
                        parts.append(-obs_dur_still)

                    all_end = np.concatenate(parts)
                    end_errors[idx] = float(np.nanmean(all_end))
                    inherited_flag[idx] = True
                    n_event_pts[idx] = int(both_any.sum()) + int(detected_not_inherited.sum())
                    n_target_evts[idx] = int(t_has_or_inh.sum())
                    # Onset, duration, peak keep their penalty-inclusive values.

                # Update active state: a point is "active" if it either
                # detected a full event or successfully inherited one.
                f_active = f_has | f_can_inherit
                t_active = t_has | t_can_inherit

            elapsed = _time.perf_counter() - t0
            n_detected = int(np.isfinite(onset_errors).sum())
            n_inherited = int(inherited_flag.sum())
            print(
                f"[METRIC:{self.name}] done in {elapsed:.2f}s, "
                f"detected={n_detected}/{n_init}, "
                f"inherited_end={n_inherited}/{n_init}",
                flush=True,
            )
            if debug_values:
                for i in range(min(debug_max_inits, n_init)):
                    tag = " [inherited]" if inherited_flag[i] else ""
                    print(
                        f"[METRIC:{self.name}] init={i}{tag} "
                        f"onset={onset_errors[i]:.2f} end={end_errors[i]:.2f} "
                        f"dur={duration_errors[i]:.2f} peak_bias={peak_bias_arr[i]:.2f} "
                        f"peak_rmse={peak_rmse_arr[i]:.2f}",
                        flush=True,
                    )

            coords = {"init_time": forecast_r["init_time"]}
            # Attach the inherited flag and station/point count as coordinates
            # on every component DataArray so they propagate through
            # .to_dataframe() into the CSV.
            inh_coord = xr.DataArray(inherited_flag, dims=("init_time",), coords=coords)
            npts_coord = xr.DataArray(n_event_pts, dims=("init_time",), coords=coords)
            ntgt_coord = xr.DataArray(n_target_evts, dims=("init_time",), coords=coords)
            result = {}
            for key, vals in [
                ("onset", onset_errors),
                ("end", end_errors),
                ("duration", duration_errors),
                ("peak_timing_bias", peak_bias_arr),
                ("peak_timing_rmse", peak_rmse_arr),
            ]:
                da = xr.DataArray(vals, dims=("init_time",), coords=coords)
                da = da.assign_coords(
                    event_inherited=inh_coord,
                    n_event_points=npts_coord,
                    n_target_events=ntgt_coord,
                )
                result[key] = da
            return result

        # =====================================================================
        # FALLBACK: no init_time (rare, e.g. single-forecast evaluation)
        # =====================================================================
        f_i, t_i, thr_i = xr.align(forecast_r, target_r, threshold, join="inner")
        if debug_values:
            print(
                f"[METRIC:{self.name}] fallback(no init_time) sizes "
                f"f={dict(f_i.sizes)} t={dict(t_i.sizes)} thr={dict(thr_i.sizes)}",
                flush=True,
            )
        time_dim = "valid_time" if "valid_time" in f_i.dims else (
            "lead_time" if "lead_time" in f_i.dims else None
        )
        if time_dim is None:
            nan_da = utils._create_nan_dataarray(self.preserve_dims)
            return {
                "onset": nan_da, "end": nan_da, "duration": nan_da,
                "peak_timing_bias": nan_da, "peak_timing_rmse": nan_da,
            }

        # Transpose so time_dim is axis 0
        non_time_dims = [d for d in f_i.dims if d != time_dim]
        dim_order = [time_dim] + non_time_dims
        f_np = np.asarray(f_i.transpose(*dim_order).values, dtype=float)
        t_np = np.asarray(
            t_i.broadcast_like(f_i).transpose(*dim_order).values, dtype=float
        )
        thr_np = np.asarray(
            thr_i.broadcast_like(f_i).transpose(*dim_order).values, dtype=float
        )

        res = _vectorized_errors_for_slice(f_np, t_np, thr_np)

        out_dim = (
            self.preserve_dims
            if isinstance(self.preserve_dims, str)
            else self.preserve_dims[0]
        )
        return {
            "onset": xr.DataArray(np.asarray([res["onset"]], dtype=float), dims=(out_dim,)),
            "end": xr.DataArray(np.asarray([res["end"]], dtype=float), dims=(out_dim,)),
            "duration": xr.DataArray(np.asarray([res["duration"]], dtype=float), dims=(out_dim,)),
            "peak_timing_bias": xr.DataArray(np.asarray([res["peak_bias"]], dtype=float), dims=(out_dim,)),
            "peak_timing_rmse": xr.DataArray(np.asarray([res["peak_rmse"]], dtype=float), dims=(out_dim,)),
        }


class ClimatologyEventTimingComposite(CompositeMetric):
    """Composite wrapper to compute event timing components once.

    Computes event detection and duration windows a single time, then serves
    onset/end/duration error child metrics from the shared precomputed results.
    """

    def __init__(
        self,
        event_kind: Literal["heat_wave", "freeze"],
        name: str | None = None,
        preserve_dims: str = "init_time",
        reduce_spatial_dims: list[str] | None = None,
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        **kwargs,
    ):
        if name is None:
            name = f"{event_kind}_timing_metrics"
        super().__init__(
            name=name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            **kwargs,
        )
        self.event_kind = event_kind
        if reduce_spatial_dims is None:
            reduce_spatial_dims = ["latitude", "longitude"]
        self.reduce_spatial_dims = reduce_spatial_dims
        self._metric_instances = [
            ClimatologyEventTimingError(
                event_kind=event_kind,
                metric_component="onset",
                preserve_dims=preserve_dims,
                reduce_spatial_dims=self.reduce_spatial_dims,
                forecast_variable=forecast_variable,
                target_variable=target_variable,
            ),
            ClimatologyEventTimingError(
                event_kind=event_kind,
                metric_component="duration",
                preserve_dims=preserve_dims,
                reduce_spatial_dims=self.reduce_spatial_dims,
                forecast_variable=forecast_variable,
                target_variable=target_variable,
            ),
            ClimatologyEventTimingError(
                event_kind=event_kind,
                metric_component="end",
                preserve_dims=preserve_dims,
                reduce_spatial_dims=self.reduce_spatial_dims,
                forecast_variable=forecast_variable,
                target_variable=target_variable,
            ),
            ClimatologyEventTimingError(
                event_kind=event_kind,
                metric_component="peak_timing_bias",
                preserve_dims=preserve_dims,
                reduce_spatial_dims=self.reduce_spatial_dims,
                forecast_variable=forecast_variable,
                target_variable=target_variable,
            ),
            ClimatologyEventTimingError(
                event_kind=event_kind,
                metric_component="peak_timing_rmse",
                preserve_dims=preserve_dims,
                reduce_spatial_dims=self.reduce_spatial_dims,
                forecast_variable=forecast_variable,
                target_variable=target_variable,
            ),
        ]

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs: Any,
    ) -> dict:
        kwargs = base_kwargs.copy()
        # Use one child instance to run shared compute once.
        shared = self._metric_instances[0].compute_all_components(
            forecast_data,
            target_data,
            debug_values=bool(base_kwargs.get("debug_heat_values", False)),
            debug_max_inits=int(base_kwargs.get("debug_heat_max_inits", 2)),
            debug_max_points=int(base_kwargs.get("debug_heat_max_points", 3)),
        )
        if bool(base_kwargs.get("debug_heat_values", False)):
            for comp_name, comp_da in shared.items():
                non_nan = int(np.isfinite(np.asarray(comp_da.values, dtype=float)).sum())
                total = int(np.asarray(comp_da.values).size)
                print(
                    f"[METRIC:{self.name}] component={comp_name} non_nan={non_nan}/{total}",
                    flush=True,
                )
        kwargs["__event_timing_components"] = shared
        return kwargs

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        raise NotImplementedError(
            "ClimatologyEventTimingComposite is a composite wrapper and should "
            "be expanded via maybe_expand_composite()."
        )


class LandfallMetric(CompositeMetric):
    """Base class for tropical cyclone landfall metrics.

    Extends CompositeMetric to compute landfalls using calc.find_landfalls,
    which utilizes land geometry and line segments based on track data to
    determine intersections.

    Can be used as a base class for custom landfall metrics, as a mixin with
    other metrics, or as a composite metric for multiple landfall metrics.

    Public methods:
        maybe_prepare_composite_kwargs: Prepare kwargs for landfall composites
    """

    def __init__(
        self,
        name: str = "landfall_metrics",
        preserve_dims: str | list[str] = ["init_time", "landfall"],
        approach: Literal["first", "next"] = "first",
        exclude_post_landfall: bool = False,
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        metrics: Optional[list[Type["LandfallMetric"]]] = None,
        *args,
        **kwargs,
    ):
        """Initialize LandfallMetric.

        Landfalls are detected using the calc.find_landfalls function, which utilizes a
        land geometry and line segments based on coordinates to determine intersections.

        Using approach, "first" will calculate the first detected landfall for an entire
        forecast, i.e. later landfalls in a multi-landfall event will not be considered.
        "next" will calculate the next landfall for each init_time.
        Using Ida as an example (case 220), "first" would only run calculations for the
        first landfall in Cuba, ignoring the later US landfall. "next" would run
        calculations for the first landfall in Cuba, then the next landfall in the US,
        etc. based on the init_time and when landfall occurred.

        Args:
            name: The name of the metric. Defaults to "landfall_metrics" for the base
                class.
            preserve_dims: The dimensions to preserve. Defaults to "init_time".
            approach: The approach to use for landfall detection. Defaults to "first".
            exclude_post_landfall: Whether to exclude post-landfall data. Defaults to
                False.
            forecast_variable: The forecast variable to use. Defaults to None.
            target_variable: The target variable to use. Defaults to None.
            metrics: A list of metrics to use as a composite. Defaults to None.
        """
        super().__init__(
            name=name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            *args,
            **kwargs,
        )
        self.approach = approach
        self.exclude_post_landfall = exclude_post_landfall
        self.metrics = metrics or []

        # If metrics provided, instantiate them
        if self.metrics is not None:
            self._metric_instances = [
                (
                    metric_cls(
                        preserve_dims=self.preserve_dims,
                        forecast_variable=self.forecast_variable,
                        target_variable=self.target_variable,
                    )
                    if isinstance(metric_cls, type)
                    else metric_cls
                )
                for metric_cls in self.metrics
            ]
        else:
            self._metric_instances = []

    def __call__(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the metric.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments
        """
        return self.compute_metric(forecast, target, **kwargs)

    def maybe_compute_landfalls(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute landfalls for a given forecast and target dataarray.

        This function computes the landfalls for a given forecast and target dataarray
        using calc.find_landfalls. Currently, this access pattern doesn't include
        passing land geometry in, but calc.find_landfalls will use NaturalEarth's 10m
        land geometry by default.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments, may include pre-computed
                forecast_landfall and target_landfall

        Returns:
            Tuple of (forecast_landfall, target_landfall). If no landfalls are found,
            returns NaN DataArrays with init_time dimension.
        """
        forecast_landfall, target_landfall = (
            kwargs.get("forecast_landfall", None),
            kwargs.get("target_landfall", None),
        )
        if forecast_landfall is not None and target_landfall is not None:
            return forecast_landfall, target_landfall

        # Check if forecast is gridded data (latitude/longitude dims) instead of track data
        # This happens when TC tracking finds no tracks and returns original gridded MSLP
        if "latitude" in forecast.dims and "longitude" in forecast.dims:
            logger.warning(
                "Landfall metric: forecast data is gridded (latitude/longitude dims present). "
                "This suggests TC tracking found no tracks. Returning NaN for landfall metrics."
            )
            # Return NaN - can't compute landfall from gridded data
            nan_landfalls = utils._create_nan_dataarray(self.preserve_dims)
            return (nan_landfalls, nan_landfalls.copy())

        # For "first" approach: get only first landfall
        # For "next" approach: get all target landfalls, then filter
        return_next_landfall = self.approach == "next"

        # Get ALL forecast landfalls (we'll select last one later for CONUS)
        # Previously this used return_next_landfall which would only get FIRST landfall
        forecast_landfalls = calc.find_landfalls(
            forecast, return_next_landfall=True  # Always get ALL landfalls
        )
        

        # If no forecast landfalls, return NaN DataArrays for both forecast and target
        if forecast_landfalls is None:
            logger.warning(
                "Landfall metric: No forecast landfalls found by calc.find_landfalls(). "
                "Forecast has dims: %s. Returning NaN for landfall metrics.",
                list(forecast.dims)
            )
            nan_landfalls = utils._create_nan_dataarray(self.preserve_dims)
            return (nan_landfalls, nan_landfalls.copy())

        # DEBUG: Save forecast tracks to CSV for visualization (only save once)
        if not hasattr(self, '_tracks_saved'):
            self._tracks_saved = True
            try:
                import pandas as pd
                from pathlib import Path
                
                # Get the forecast source name for per-model CSV filenames
                forecast_source = kwargs.get('forecast_source', 'unknown')
                safe_name = forecast_source.replace(" ", "_").replace("/", "_")
                output_path = Path(
                    f"/huge/users/larissa/ExtremeWeatherBench/run/forecast_tracks_{safe_name}.csv"
                )
                lock_path = output_path.with_suffix(output_path.suffix + ".lock")

                # If already written by another metric/worker, skip both save and prints.
                should_write = not output_path.exists()

                # Best-effort cross-process lock to avoid duplicate work/log spam.
                lock_handle = None
                acquired_lock = False
                if should_write:
                    try:
                        lock_handle = lock_path.open("x")
                        lock_handle.write("track_csv_write_lock\n")
                        lock_handle.flush()
                        acquired_lock = True
                    except FileExistsError:
                        # Another worker is already writing this model's track CSV.
                        acquired_lock = False
                        should_write = False

                if should_write and acquired_lock:
                    print(f"\n  [Track CSV DEBUG] Starting track save...")
                    print(f"    forecast type: {type(forecast).__name__}, dims: {list(forecast.dims)}")
                    print(f"    coords: {list(forecast.coords)}", flush=True)

                    # The forecast DataArray has latitude/longitude as coordinates.
                    # For TC tracks we expect lead_time/valid_time dims.
                    if 'latitude' in forecast.coords and 'longitude' in forecast.coords:
                        track_records = []

                        if 'lead_time' in forecast.dims and 'valid_time' in forecast.dims:
                            print(f"    Detected TC track structure (lead_time, valid_time)")
                            for lt_idx in range(len(forecast.lead_time)):
                                for vt_idx in range(len(forecast.valid_time)):
                                    try:
                                        point = forecast.isel(lead_time=lt_idx, valid_time=vt_idx)
                                        lat = float(point.latitude.values)
                                        lon = float(point.longitude.values)
                                        vt = pd.Timestamp(point.valid_time.values)
                                        lt = point.lead_time.values
                                        lt_hours = float(lt / np.timedelta64(1, 'h'))
                                        val = float(point.values)

                                        if not np.isnan(lat) and not np.isnan(lon):
                                            record = {
                                                'valid_time': vt,
                                                'lead_time_hours': lt_hours,
                                                'latitude': lat,
                                                'longitude': lon,
                                                'value': val,
                                            }
                                            if 'init_time' in point.coords:
                                                record['init_time'] = pd.Timestamp(point.init_time.values)
                                            track_records.append(record)
                                    except Exception:
                                        continue

                            print(f"    Extracted {len(track_records)} track points")

                        if track_records:
                            track_df = pd.DataFrame(track_records)
                            track_df.to_csv(output_path, index=False)
                            print(f"  [Track CSV DEBUG] ✓ Saved to {output_path}")
                            if 'init_time' in track_df.columns:
                                print(f"    {track_df['init_time'].nunique()} init times")
                            print(f"    Lat range: [{track_df['latitude'].min():.1f}, {track_df['latitude'].max():.1f}]")
                            print(f"    Lon range: [{track_df['longitude'].min():.1f}, {track_df['longitude'].max():.1f}]")
                        else:
                            print(f"  [Track CSV DEBUG] No valid track points extracted")
                    else:
                        print(f"  [Track CSV DEBUG] lat/lon not in forecast.coords, skipping")
            except Exception as e:
                print(f"  [Track CSV DEBUG] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    if 'lock_handle' in locals() and lock_handle is not None:
                        lock_handle.close()
                    if (
                        'acquired_lock' in locals()
                        and acquired_lock
                        and 'lock_path' in locals()
                        and lock_path.exists()
                    ):
                        lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # DEBUG: Show forecast landfall structure
        if "landfall" in forecast_landfalls.dims:
            n_landfalls = len(forecast_landfalls.landfall)
            n_inits = len(forecast_landfalls.init_time) if 'init_time' in forecast_landfalls.dims else 1
            print(f"\n  [LandfallMetric DEBUG] Forecast landfalls structure:")
            print(f"    {n_landfalls} landfall slots × {n_inits} init_times")
            
            # Count valid entries per landfall slot
            for i in range(n_landfalls):
                lf = forecast_landfalls.isel(landfall=i)
                lat_vals = lf.latitude.values
                n_valid = int(np.sum(~np.isnan(lat_vals))) if hasattr(lat_vals, '__len__') else (0 if np.isnan(float(lat_vals)) else 1)
                if n_valid > 0:
                    print(f"    Landfall slot {i}: {n_valid}/{n_inits} valid, lat range [{np.nanmin(lat_vals):.1f}, {np.nanmax(lat_vals):.1f}]")
                else:
                    print(f"    Landfall slot {i}: {n_valid}/{n_inits} valid (all NaN)")
            
            # NOTE: Forecast landfall selection deferred until after target is loaded
            # (see geographic distance matching below)
        else:
            print(f"\n  [LandfallMetric DEBUG] Single forecast landfall (no landfall dim)")

        # Get all target landfalls (always request all, then select last/appropriate one)
        target_landfalls_pre_init = calc.find_landfalls(
            target, return_next_landfall=True  # Get ALL landfalls, not just first
        )
        

        # If no target landfalls, return NaN DataArrays for both forecast and target
        if target_landfalls_pre_init is None:
            print(f"  [LandfallMetric DEBUG] No target landfalls found!")
            nan_landfalls = utils._create_nan_dataarray(self.preserve_dims)
            return (nan_landfalls, nan_landfalls.copy())

        # DEBUG: Show all target landfalls
        if "landfall" in target_landfalls_pre_init.dims:
            n_target = len(target_landfalls_pre_init.landfall)
            print(f"\n  [LandfallMetric DEBUG] Found {n_target} target landfall(s):")
            for i in range(min(n_target, 10)):  # Show up to 10
                lf = target_landfalls_pre_init.isel(landfall=i)
                lat = float(lf.latitude.values) if 'latitude' in lf.coords else "?"
                lon = float(lf.longitude.values) if 'longitude' in lf.coords else "?"
                vt = str(lf.valid_time.values)[:19] if 'valid_time' in lf.coords else "?"
                print(f"    [{i}] {vt} at ({lat:.1f}°N, {lon:.1f}°E)")
            print(f"  → Will track ALL {n_target} landfall(s) separately")
            
            # Match forecast landfalls to EACH target landfall by geographic distance
            if "landfall" in forecast_landfalls.dims and 'init_time' in forecast_landfalls.dims:
                print(f"  → Matching forecast landfalls by distance to each target...")
                
                def haversine_distance(lat1, lon1, lat2, lon2):
                    """Calculate distance in km between two lat/lon points."""
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371  # Earth radius in km
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    return R * c
                
                # Match for each target landfall
                all_matched = []
                for target_idx in range(n_target):
                    target_lf = target_landfalls_pre_init.isel(landfall=target_idx)
                    target_lat = float(target_lf.latitude.values)
                    target_lon = float(target_lf.longitude.values)
                    target_time = target_lf.valid_time.values
                    
                    print(f"    Target landfall {target_idx}: ({target_lat:.1f}°N, {target_lon:.1f}°E) at {str(target_time)[:19]}")

                    selected = []
                    for init_idx in range(n_inits):
                        init_time = forecast_landfalls.init_time.values[init_idx]
                        
                        # Template NaN landfall point used when a match is not valid.
                        # Set both coords and data to NaN so all landfall metrics stay
                        # consistent (displacement/time/intensity).
                        nan_point = forecast_landfalls.isel(init_time=init_idx, landfall=0).copy()
                        nan_point = nan_point * np.nan
                        nan_point = nan_point.assign_coords(
                            latitude=np.nan,
                            longitude=np.nan,
                            valid_time=np.datetime64('NaT', 'ns')
                        )
                        
                        # Only consider forecasts initialized before this landfall
                        if init_time >= target_time:
                            # Forecast initialized after landfall - use NaN
                            selected.append(nan_point)
                            continue
                        
                        # Find forecast landfall closest to this target
                        best_distance = float('inf')
                        best_lf_idx = 0
                        for lf_idx in range(n_landfalls):
                            lf = forecast_landfalls.isel(init_time=init_idx, landfall=lf_idx)
                            fc_lat = float(lf.latitude.values)
                            fc_lon = float(lf.longitude.values)
                            
                            if not np.isnan(fc_lat) and not np.isnan(fc_lon):
                                distance = haversine_distance(fc_lat, fc_lon, target_lat, target_lon)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_lf_idx = lf_idx

                        # No valid forecast landfall found for this init/target pair.
                        # Keep this point as NaN rather than defaulting to landfall=0.
                        if not np.isfinite(best_distance):
                            selected.append(nan_point)
                            continue

                        sel = forecast_landfalls.isel(init_time=init_idx, landfall=best_lf_idx)
                        # Drop 'landfall' coordinate entirely - it's now a scalar after isel
                        if 'landfall' in sel.coords:
                            sel = sel.drop_vars('landfall', errors='ignore')
                        selected.append(sel)
                    
                    # Concat once per target landfall (not per init_time).
                    matched_for_target = xr.concat(
                        selected,
                        dim='init_time',
                        coords='minimal',
                        compat='override',
                    )
                    # Guard against duplicated init_time labels coming from upstream
                    # forecast data assembly; xarray alignment requires uniqueness.
                    if "init_time" in matched_for_target.coords:
                        init_index = pd.Index(matched_for_target["init_time"].values)
                        if init_index.has_duplicates:
                            matched_for_target = matched_for_target.groupby("init_time").first()
                    # Make sure landfall coordinate is gone before stacking
                    if 'landfall' in matched_for_target.coords:
                        matched_for_target = matched_for_target.drop_vars('landfall', errors='ignore')
                    all_matched.append(matched_for_target)
                
                # Stack all target landfalls along a new dimension
                forecast_landfalls = xr.concat(
                    all_matched,
                    dim='landfall',
                    join='outer',
                    coords='minimal',
                    compat='override',
                )
                
                print(f"    Matched {n_target} target landfall(s) across {n_inits} init times")
                print(f"    Result shape: {dict(forecast_landfalls.sizes)}")
        else:
            print(f"\n  [LandfallMetric DEBUG] Single target landfall")
            lat = float(target_landfalls_pre_init.latitude.values) if 'latitude' in target_landfalls_pre_init.coords else "?"
            lon = float(target_landfalls_pre_init.longitude.values) if 'longitude' in target_landfalls_pre_init.coords else "?"
            print(f"    lat={lat:.1f}, lon={lon:.1f}")

        if return_next_landfall:
            # Find next target landfall for each init_time
            # Note: This now works with multi-landfall structure
            target_landfalls = calc.find_next_landfall_for_init_time(
                forecast_landfalls, target_landfalls_pre_init
            )
        else:
            # Keep ALL target landfalls (don't select just last one)
                target_landfalls = target_landfalls_pre_init
            
            # Filter forecasts to only those initialized before each landfall
            # This is handled above in the matching loop (NaN for late inits)
        
        return forecast_landfalls, target_landfalls

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Computes the landfalls once and adds them to kwargs to avoid recomputing when
        used as a composite metric.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

        Returns:
            Dictionary of kwargs including transformed_manager.
        """
        kwargs = base_kwargs.copy()

        if self.is_composite() and len(self._metric_instances) > 1:
            kwargs["forecast_landfall"], kwargs["target_landfall"] = (
                self.maybe_compute_landfalls(
                    forecast=forecast_data, target=target_data, **base_kwargs
                )
            )

        kwargs["preserve_dims"] = self.preserve_dims

        return kwargs

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute metric (not supported for LandfallMetric base)

        LandfallMetric must be subclassed (like LandfallDisplacement,
        LandfallTimeMeanError)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments
        """
        raise NotImplementedError(
            "LandfallMetric._compute_metric must be implemented "
            "by subclasses (LandfallDisplacement, LandfallTimeMeanError, etc.) or use "
            "LandfallMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class SpatialDisplacement(BaseMetric):
    """Compute spatial displacement between forecast and target patterns.

    Extends BaseMetric to compute great circle distance between centers of
    mass of forecast and target spatial patterns. Useful for atmospheric
    rivers and similar spatial features.
    """

    def __init__(
        self,
        name: str = "spatial_displacement",
        **kwargs: Any,
    ):
        """Initialize the Spatial Displacement metric.

        Args:
            name: The name of the metric. Defaults to
                "spatial_displacement".
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute spatial displacement.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            The spatial displacement between the forecast and target as a DataArray.
        """

        def center_of_mass_ufunc(data):
            """ufunc tooling to calculate the center of mass of a 2D array, returning
            a tuple of the latitude and longitude indices, or np.nan tuple if no
            non-zero values are present.
            """
            if (data > 0).any():
                return ndimage.center_of_mass(data)
            else:
                return (np.nan, np.nan)

        target_lat_idx, target_lon_idx = xr.apply_ufunc(
            center_of_mass_ufunc,
            target,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="allowed",
        )

        # Process target coordinates
        target_lat_idx = np.round(target_lat_idx)
        target_lon_idx = np.round(target_lon_idx)
        target_lat_coords, target_lon_coords = utils.idx_to_coords(
            target_lat_idx,
            target_lon_idx,
            target.latitude.values,
            target.longitude.values,
        )
        target_coordinates = np.array([target_lat_coords, target_lon_coords])

        # Process forecast coordinates
        forecast_lat_idx, forecast_lon_idx = xr.apply_ufunc(
            center_of_mass_ufunc,
            forecast,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="allowed",
        )
        forecast_lat_idx = np.round(forecast_lat_idx)
        forecast_lon_idx = np.round(forecast_lon_idx)
        forecast_lat_coords, forecast_lon_coords = utils.idx_to_coords(
            forecast_lat_idx,
            forecast_lon_idx,
            forecast.latitude.values,
            forecast.longitude.values,
        )
        forecast_coordinates = np.array([forecast_lat_coords, forecast_lon_coords])

        # Calculate haversine distance
        distance = calc.haversine_distance(forecast_coordinates, target_coordinates)

        # Create DataArray with all dimensions
        result = xr.DataArray(
            distance,
            coords={"lead_time": forecast.lead_time, "valid_time": forecast.valid_time},
            dims=["lead_time", "valid_time"],
            name="spatial_displacement",
        )

        # Reduce over non-preserved dimensions (valid_time) by taking mean
        time_dims_to_reduce = [
            dim for dim in result.dims if dim not in self.preserve_dims
        ]
        if time_dims_to_reduce:
            result = result.mean(dim=time_dims_to_reduce)

        return result


class LandfallDisplacement(LandfallMetric):
    """Compute distance between forecast and target landfall positions.

    Extends LandfallMetric to calculate the spatial distance between forecast
    and target landfall positions, defaulting to kilometers.
    """

    def __init__(
        self,
        name: str = "landfall_displacement",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Displacement metric.

        Args:
            name: The name of the metric. Defaults to
                "landfall_displacement".
            *args: Additional positional arguments passed to LandfallMetric.
            **kwargs: Additional keyword arguments passed to LandfallMetric.
        """
        super().__init__(name, *args, **kwargs)
        self.units = kwargs.get("units", "km")

    def calculate_displacement(
        self,
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
        units: Literal["km", "kilometers", "deg", "degrees"] = "km",
    ) -> xr.DataArray:
        """Calculate the distance between two landfall points in kilometers or degrees.

        Handles multi-landfall tracking: forecast and target can have 'landfall' dimension.
        For each target landfall, computes distance to matched forecast landfall.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray (dims: landfall, init_time)
            target_landfall: Target landfall xarray DataArray (dims: landfall)
            units: The units to use for the distance. Defaults to "km"
        Returns:
            Distance in the specified units as xarray DataArray (dims: landfall, init_time)
        """
        # Check if we have multi-landfall structure
        has_landfall_dim = "landfall" in forecast_landfall.dims and "landfall" in target_landfall.dims
        
        if not has_landfall_dim:
            # Legacy single-landfall path
            t_lat = float(target_landfall.coords["latitude"].values)
            t_lon = float(target_landfall.coords["longitude"].values)
        
            if np.isnan(t_lat) or np.isnan(t_lon):
                return utils._create_nan_dataarray(self.preserve_dims)
            
            if "init_time" in forecast_landfall.dims:
                init_times = forecast_landfall.coords["init_time"].values
            elif "init_time" in forecast_landfall.coords:
                init_times = [forecast_landfall.coords["init_time"].values]
            else:
                return utils._create_nan_dataarray(self.preserve_dims)

            distances = []
            valid_init_times = []
            for init_time in init_times:
                if "init_time" in forecast_landfall.dims:
                    f_data = forecast_landfall.sel(init_time=init_time)
                else:
                    f_data = forecast_landfall
                
                f_lat = float(f_data.coords["latitude"].values)
                f_lon = float(f_data.coords["longitude"].values)

                if np.isnan(f_lat) or np.isnan(f_lon):
                    continue
                
                dist = calc.haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units=units
                )
                distances.append(
                    float(dist.item()) if hasattr(dist, "item") else float(dist)
                )
                valid_init_times.append(init_time)

            if not distances:
                return utils._create_nan_dataarray(self.preserve_dims)

            return xr.DataArray(
                distances,
                dims=["init_time"],
                coords={"init_time": valid_init_times},
            )
        
        # Multi-landfall path: compute distance for each landfall separately.
        # Forecast/target landfall counts can differ; only compare shared indices and
        # leave unmatched forecast landfalls as NaN.
        n_forecast_landfalls = len(forecast_landfall.landfall)
        n_target_landfalls = len(target_landfall.landfall)
        n_shared_landfalls = min(n_forecast_landfalls, n_target_landfalls)
        n_inits = len(forecast_landfall.init_time) if 'init_time' in forecast_landfall.dims else 1
        
        print(f"[DEBUG LandfallDisplacement] forecast dims: {forecast_landfall.dims}, target dims: {target_landfall.dims}")
        print(
            f"  n_forecast_landfalls={n_forecast_landfalls}, "
            f"n_target_landfalls={n_target_landfalls}, "
            f"n_shared_landfalls={n_shared_landfalls}, n_inits={n_inits}"
        )
        
        # Initialize distance array
        distances = np.full((n_forecast_landfalls, n_inits), np.nan)
        
        for lf_idx in range(n_shared_landfalls):
            # Get target for this landfall
            target_lf = target_landfall.isel(landfall=lf_idx)
            
            # Debug: Show target structure
            if lf_idx == 0:
                print(f"[DEBUG] target_lf structure: {target_lf}")
                print(f"[DEBUG] target_lf coords: {list(target_lf.coords.keys())}")
                print(f"[DEBUG] target_lf dims: {target_lf.dims}")
            
            t_lat = float(target_lf.coords["latitude"].values)
            t_lon = float(target_lf.coords["longitude"].values)
            
            if lf_idx == 0:
                print(f"[DEBUG] Target landfall 0: lat={t_lat}, lon={t_lon}")
            
            if np.isnan(t_lat) or np.isnan(t_lon):
                print(f"[DEBUG] Target landfall {lf_idx}: NaN coordinates (lat={t_lat}, lon={t_lon})")
                continue  # Leave as NaN
            
            # Compute distance for each init_time
            for init_idx in range(n_inits):
                forecast_lf = forecast_landfall.isel(landfall=lf_idx, init_time=init_idx)
                
                # Debug: Show forecast structure for first landfall/init
                if lf_idx == 0 and init_idx == 0:
                    print(f"[DEBUG] forecast_lf structure: {forecast_lf}")
                    print(f"[DEBUG] forecast_lf coords: {list(forecast_lf.coords.keys())}")
                
                f_lat = float(forecast_lf.coords["latitude"].values)
                f_lon = float(forecast_lf.coords["longitude"].values)
                
                if lf_idx == 0 and init_idx == 0:
                    print(f"[DEBUG] Forecast landfall 0, init 0: lat={f_lat}, lon={f_lon}")
                
                if np.isnan(f_lat) or np.isnan(f_lon):
                    if lf_idx == 0 and init_idx == 0:
                        print(f"[DEBUG] Forecast NaN for lf={lf_idx}, init={init_idx}")
                    continue  # Leave as NaN
                
                dist = calc.haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units=units
                )
                distances[lf_idx, init_idx] = float(dist.item()) if hasattr(dist, "item") else float(dist)
                
                if lf_idx == 0 and init_idx == 0:
                    print(f"[DEBUG] Computed distance for lf=0, init=0: {distances[lf_idx, init_idx]} km")
        
        print(f"[DEBUG LandfallDisplacement] Computed distances:")
        print(f"  shape: {distances.shape}")
        print(f"  non-NaN count: {np.sum(~np.isnan(distances))}")
        print(f"  sample values: {distances[~np.isnan(distances)][:5] if np.any(~np.isnan(distances)) else 'all NaN'}")
        
        # Build result with landfall and init_time dimensions
        result = xr.DataArray(
            distances,
            dims=["landfall", "init_time"],
            coords={
                "landfall": forecast_landfall.landfall,
                "init_time": forecast_landfall.init_time,
            }
        )
        
        print(f"[DEBUG LandfallDisplacement] Returning result with dims: {result.dims}, shape: {result.shape}")
        
        return result

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall displacement metric."""
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        # Forecast needs init_time, target doesn't (it's observed)
        print(f"[DEBUG LandfallDisplacement] Checking validity...")
        print(f"  forecast_landfall dims: {forecast_landfall.dims if hasattr(forecast_landfall, 'dims') else 'NO DIMS'}")
        print(f"  target_landfall dims: {target_landfall.dims if hasattr(target_landfall, 'dims') else 'NO DIMS'}")
        
        forecast_valid = utils.is_valid_landfall(forecast_landfall, require_init_time=True)
        target_valid = utils.is_valid_landfall(target_landfall, require_init_time=False)
        
        print(f"  forecast_valid={forecast_valid}, target_valid={target_valid}")
        
        if not forecast_valid or not target_valid:
            print(f"  → Returning NaN (invalid landfall)")
            return utils._create_nan_dataarray(self.preserve_dims)
        return self.calculate_displacement(
            forecast_landfall,
            target_landfall,
            units=self.units,
        )


class LandfallTimeMeanError(LandfallMetric):
    """Compute mean error between forecast and target landfall times.

    Extends LandfallMetric to calculate timing difference. Positive values
    indicate forecast landfall is later than target; negative values indicate
    forecast landfall is earlier than target.
    """

    def __init__(
        self,
        name: str = "landfall_time_me",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Time Mean Error metric.

        Args:
            name: The name of the metric. Defaults to "landfall_time_me".
            *args: Additional positional arguments passed to LandfallMetric.
            **kwargs: Additional keyword arguments passed to LandfallMetric.
        """
        super().__init__(name, *args, **kwargs)

    def calculate_time_difference(
        self,
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate the time difference between two landfall points in hours.

        Handles multi-landfall tracking: forecast and target can have 'landfall' dimension.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray.
            target_landfall: Target landfall xarray DataArray.

        Returns:
            Time difference in hours (forecast_landfall - target_landfall)
            as xarray DataArray with dimensions matching input.
        """
        # Check if we have multi-landfall structure
        has_landfall_dim = (
            "landfall" in forecast_landfall.dims and "landfall" in target_landfall.dims
        )

        if not has_landfall_dim:
            # Legacy single-landfall path
            t_time = target_landfall.coords["valid_time"].values

            if "init_time" in forecast_landfall.dims:
                init_times = forecast_landfall.coords["init_time"].values
            elif "init_time" in forecast_landfall.coords:
                init_times = [forecast_landfall.coords["init_time"].values]
            else:
                return utils._create_nan_dataarray(self.preserve_dims)

            time_diffs = []
            valid_init_times = []
            for init_time in init_times:
                if "init_time" in forecast_landfall.dims:
                    f_data = forecast_landfall.sel(init_time=init_time)
                else:
                    f_data = forecast_landfall

                f_time = f_data.coords["valid_time"].values
                if pd.isna(f_time) or pd.isna(t_time):
                    continue

                time_diff = (f_time - t_time) / np.timedelta64(1, "h")
                time_diffs.append(float(time_diff))
                valid_init_times.append(init_time)

            if not time_diffs:
                return utils._create_nan_dataarray(self.preserve_dims)

            return xr.DataArray(
                time_diffs,
                dims=["init_time"],
                coords={"init_time": valid_init_times},
            )

        # Multi-landfall path
        if "init_time" in forecast_landfall.dims:
            init_times = forecast_landfall.coords["init_time"].values
        elif "init_time" in forecast_landfall.coords:
            init_times = [forecast_landfall.coords["init_time"].values]
        else:
            return utils._create_nan_dataarray(self.preserve_dims)

        n_forecast_landfalls = len(forecast_landfall.landfall)
        n_target_landfalls = len(target_landfall.landfall)
        n_shared_landfalls = min(n_forecast_landfalls, n_target_landfalls)
        n_inits = len(init_times)
        time_diffs = np.full((n_forecast_landfalls, n_inits), np.nan)

        for lf_idx in range(n_shared_landfalls):
            target_lf = target_landfall.isel(landfall=lf_idx)
            t_time = target_lf.coords["valid_time"].values

            for init_idx in range(n_inits):
                if "init_time" in forecast_landfall.dims:
                    forecast_lf = forecast_landfall.isel(landfall=lf_idx, init_time=init_idx)
                else:
                    forecast_lf = forecast_landfall.isel(landfall=lf_idx)
                f_time = forecast_lf.coords["valid_time"].values

                # Skip if either timestamp is NaT/NaN
                if pd.isna(f_time) or pd.isna(t_time):
                    continue

                time_diff = (f_time - t_time) / np.timedelta64(1, "h")
                time_diffs[lf_idx, init_idx] = float(time_diff)

        result = xr.DataArray(
            time_diffs,
            dims=["landfall", "init_time"],
            coords={
                "landfall": forecast_landfall.landfall,
                "init_time": init_times,
            },
        )

        if "landfall_id" in forecast_landfall.coords:
            result = result.assign_coords(
                landfall_id=("landfall", forecast_landfall.landfall_id.values)
            )

        return result

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall time metric."""
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall, require_init_time=True
        ) or not utils.is_valid_landfall(target_landfall, require_init_time=False):
            return utils._create_nan_dataarray(self.preserve_dims)
        return self.calculate_time_difference(forecast_landfall, target_landfall)


class LandfallIntensityMeanAbsoluteError(LandfallMetric, MeanAbsoluteError):
    """Compute MAE of forecast and target intensity at landfall.

    Extends both LandfallMetric and MeanAbsoluteError to calculate mean
    absolute error between forecast and target intensity at landfall time.

    The intensity variable is determined by forecast_variable and
    target_variable. For multiple intensity variables, create separate metric
    instances for each variable.
    """

    def __init__(
        self,
        name: str = "landfall_intensity_mae",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Intensity Mean Absolute Error metric.

        Args:
            name: The name of the metric. Defaults to
                "landfall_intensity_mae".
            *args: Additional positional arguments passed to parent classes.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall intensity metric.
        
        Handles multi-landfall tracking: forecast and target can have 'landfall' dimension.
        """
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall, require_init_time=True
        ) or not utils.is_valid_landfall(target_landfall, require_init_time=False):
            return utils._create_nan_dataarray(self.preserve_dims)

        # Check if we have multi-landfall structure
        has_landfall_dim = (
            "landfall" in forecast_landfall.dims and "landfall" in target_landfall.dims
        )

        if not has_landfall_dim:
            # Legacy single-landfall path
            target_value = float(target_landfall.values)
            if np.isnan(target_value):
                return utils._create_nan_dataarray(self.preserve_dims)

            intensity_errors = np.abs(forecast_landfall.values - target_value)

            if "init_time" in forecast_landfall.dims:
                init_times = forecast_landfall.coords["init_time"].values
            elif "init_time" in forecast_landfall.coords:
                init_times = [forecast_landfall.coords["init_time"].values]
            else:
                return utils._create_nan_dataarray(self.preserve_dims)

            return xr.DataArray(
                intensity_errors,
                dims=["init_time"],
                coords={"init_time": init_times},
            )

        # Multi-landfall path
        if "init_time" in forecast_landfall.dims:
            init_times = forecast_landfall.coords["init_time"].values
        elif "init_time" in forecast_landfall.coords:
            init_times = [forecast_landfall.coords["init_time"].values]
        else:
            return utils._create_nan_dataarray(self.preserve_dims)

        n_forecast_landfalls = len(forecast_landfall.landfall)
        n_target_landfalls = len(target_landfall.landfall)
        n_shared_landfalls = min(n_forecast_landfalls, n_target_landfalls)
        n_inits = len(init_times)
        intensity_errors = np.full((n_forecast_landfalls, n_inits), np.nan)

        for lf_idx in range(n_shared_landfalls):
            target_lf = target_landfall.isel(landfall=lf_idx)
            target_value = float(target_lf.values)

            if np.isnan(target_value):
                continue

            for init_idx in range(n_inits):
                if "init_time" in forecast_landfall.dims:
                    forecast_lf = forecast_landfall.isel(landfall=lf_idx, init_time=init_idx)
                else:
                    forecast_lf = forecast_landfall.isel(landfall=lf_idx)
                forecast_value = float(forecast_lf.values)

                if np.isnan(forecast_value):
                    continue

                intensity_errors[lf_idx, init_idx] = np.abs(forecast_value - target_value)

        result = xr.DataArray(
            intensity_errors,
            dims=["landfall", "init_time"],
            coords={
                "landfall": forecast_landfall.landfall,
                "init_time": init_times,
            },
        )

        if "landfall_id" in forecast_landfall.coords:
            result = result.assign_coords(
                landfall_id=("landfall", forecast_landfall.landfall_id.values)
            )

        return result


class LandfallIntensityRootMeanSquaredError(LandfallIntensityMeanAbsoluteError):
    """Compute RMSE of forecast and target intensity at landfall.

    Uses the same underlying landfall extraction and pairing logic as
    ``LandfallIntensityMeanAbsoluteError``, but reports the root mean squared
    error quantity under the metric name ``landfall_intensity_rmse``.
    """

    def __init__(
        self,
        name: str = "landfall_intensity_rmse",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Base class returns absolute intensity errors at landfall. For a
        # pointwise landfall comparison, RMSE reduces to sqrt(err^2) per sample.
        result = super()._compute_metric(forecast, target, **kwargs)
        if isinstance(result, xr.DataArray):
            return np.sqrt(result**2)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Track‐error decomposition metrics (along‐track / cross‐track / total)
# ═══════════════════════════════════════════════════════════════════════════════


def _observed_heading(
    target: xr.DataArray,
) -> xr.DataArray:
    """Compute observed track heading at each valid_time using central differences.

    For interior points, the heading is the azimuth from the previous to the
    next observed position (central difference).  At endpoints the forward or
    backward difference is used instead.

    Args:
        target: Target DataArray with ``valid_time`` dim and ``latitude``,
            ``longitude`` coords indexed by ``valid_time``.

    Returns:
        DataArray of heading values in **radians**, indexed by ``valid_time``.
    """
    vts = target.valid_time.values
    lats = target.latitude.values.astype(float)
    lons = target.longitude.values.astype(float)

    n = len(vts)
    headings = np.full(n, np.nan)

    for i in range(n):
        if n == 1:
            headings[i] = 0.0
        elif i == 0:
            headings[i] = calc.forward_azimuth(lats[0], lons[0], lats[1], lons[1])
        elif i == n - 1:
            headings[i] = calc.forward_azimuth(
                lats[n - 2], lons[n - 2], lats[n - 1], lons[n - 1]
            )
        else:
            headings[i] = calc.forward_azimuth(
                lats[i - 1], lons[i - 1], lats[i + 1], lons[i + 1]
            )

    return xr.DataArray(
        headings, dims=["valid_time"], coords={"valid_time": vts}, name="heading"
    )


def _compute_track_error_decomposition(
    forecast: xr.DataArray,
    target: xr.DataArray,
    units: str = "km",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Shared computation for along-track, cross-track, and total track error.

    Matches forecast and target positions at common ``valid_time`` values,
    computes the observed heading from the target track, and decomposes the
    position error into along‐track and cross‐track components.

    Args:
        forecast: Forecast DataArray with dims ``[lead_time, valid_time]`` and
            ``latitude`` / ``longitude`` coords.
        target: Target DataArray with dim ``[valid_time]`` and ``latitude`` /
            ``longitude`` coords.
        units: Distance units (``"km"`` or ``"degrees"``).

    Returns:
        Tuple of ``(ate, cte, total)`` DataArrays with same dims as *forecast*.
        ``ate`` is positive when the forecast is ahead (too fast).
        ``cte`` is positive when the forecast is to the right of the observed
        heading.
        ``total`` is the unsigned great‐circle distance.
    """
    import pandas as pd

    # Identify common valid_times
    fc_vts = pd.DatetimeIndex(forecast.valid_time.values)
    tgt_vts = pd.DatetimeIndex(target.valid_time.values)
    common_vts = fc_vts.intersection(tgt_vts)

    # Pre-compute observed heading along the target track
    heading_da = _observed_heading(target)

    # Allocate output arrays (same shape as forecast)
    ate_vals = np.full(forecast.shape, np.nan)
    cte_vals = np.full(forecast.shape, np.nan)
    tot_vals = np.full(forecast.shape, np.nan)

    # Build fast lookup: valid_time → index in the target arrays
    tgt_vt_to_idx = {pd.Timestamp(vt): i for i, vt in enumerate(tgt_vts)}

    fc_lats = forecast.latitude.values.astype(float)
    fc_lons = forecast.longitude.values.astype(float)
    tgt_lats = target.latitude.values.astype(float)
    tgt_lons = target.longitude.values.astype(float)
    headings = heading_da.values

    for vt_idx, vt in enumerate(fc_vts):
        if vt not in common_vts:
            continue
        tgt_idx = tgt_vt_to_idx.get(vt)
        if tgt_idx is None:
            continue

        obs_lat = tgt_lats[tgt_idx]
        obs_lon = tgt_lons[tgt_idx]
        heading = headings[tgt_idx]

        if np.isnan(obs_lat) or np.isnan(obs_lon) or np.isnan(heading):
            continue

        for lt_idx in range(forecast.sizes["lead_time"]):
            fc_lat = fc_lats[lt_idx, vt_idx]
            fc_lon = fc_lons[lt_idx, vt_idx]

            if np.isnan(fc_lat) or np.isnan(fc_lon):
                continue

            ate, cte = calc.along_cross_track_errors(
                fc_lat, fc_lon, obs_lat, obs_lon, heading, units=units
            )
            tot = calc.haversine_distance(
                [obs_lat, obs_lon], [fc_lat, fc_lon], units=units
            )

            ate_vals[lt_idx, vt_idx] = float(ate)
            cte_vals[lt_idx, vt_idx] = float(cte)
            tot_vals[lt_idx, vt_idx] = float(tot)

    # Preserve all coordinates from forecast, not just lead_time and valid_time
    coords = {
        "lead_time": forecast.lead_time.values,
        "valid_time": forecast.valid_time.values,
    }
    # Add any additional coordinates (like init_time, latitude, longitude)
    for coord_name in forecast.coords:
        if coord_name not in coords:
            coords[coord_name] = forecast.coords[coord_name]
    
    dims = ["lead_time", "valid_time"]

    ate_da = xr.DataArray(ate_vals, dims=dims, coords=coords, name="along_track_error")
    cte_da = xr.DataArray(cte_vals, dims=dims, coords=coords, name="cross_track_error")
    tot_da = xr.DataArray(tot_vals, dims=dims, coords=coords, name="total_track_error")

    return ate_da, cte_da, tot_da


class AlongTrackError(BaseMetric):
    """Along-track component of tropical cyclone position error.

    Projects the great‐circle displacement between forecast and observed
    positions onto the axis *parallel* to the observed track heading.

    Positive values indicate the forecast storm is **ahead** of the observed
    position (propagating too fast); negative values indicate it is **behind**
    (too slow).  This is a standard metric for diagnosing TC propagation
    speed biases.

    The output is indexed by ``lead_time`` (default ``preserve_dims``),
    averaging over all initializations for each lead time.

    Units default to **km**.
    """

    def __init__(
        self,
        name: str = "along_track_error",
        preserve_dims: list = None,
        units: str = "km",
        **kwargs: Any,
    ):
        """Initialize AlongTrackError.

        Args:
            name: Metric name.  Defaults to ``"along_track_error"``.
            preserve_dims: Dimension(s) to keep.  Defaults to ``["lead_time", "valid_time"]``
                to preserve init_time coordinate.
            units: ``"km"`` or ``"degrees"``.  Defaults to ``"km"``.
            **kwargs: Passed to ``BaseMetric.__init__``.
        """
        if preserve_dims is None:
            preserve_dims = ["lead_time", "valid_time"]
        super().__init__(name, preserve_dims=preserve_dims, **kwargs)
        self.units = units

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute along-track error.

        Args:
            forecast: Forecast DataArray (TC track with lat/lon coords).
            target: Target DataArray (observed track with lat/lon coords).

        Returns:
            Along-track error DataArray, reduced to ``preserve_dims``.
        """
        print(f"\n[DEBUG AlongTrackError] Input forecast:")
        print(f"  dims: {forecast.dims}")
        print(f"  coords: {list(forecast.coords)}")
        if "init_time" in forecast.coords:
            print(f"  init_time shape: {forecast.init_time.shape}")
            print(f"  init_time sample: {forecast.init_time.values.flat[:3]}")
        else:
            print(f"  ⚠️ init_time NOT in forecast coords")
        
        ate, _, _ = _compute_track_error_decomposition(
            forecast, target, units=self.units
        )
        
        print(f"[DEBUG AlongTrackError] After decomposition:")
        print(f"  ate.dims: {ate.dims}")
        print(f"  ate.coords: {list(ate.coords)}")
        if "init_time" in ate.coords:
            print(f"  init_time still present in ate")
        else:
            print(f"  ⚠️ init_time LOST after decomposition")
        
        # Reduce over non-preserved dimensions
        preserve_dims = self.preserve_dims if isinstance(self.preserve_dims, list) else [self.preserve_dims]
        dims_to_reduce = [d for d in ate.dims if d not in preserve_dims]
        if dims_to_reduce:
            print(f"  Reducing over dims: {dims_to_reduce}")
            ate = ate.mean(dim=dims_to_reduce, skipna=True)
        else:
            print(f"  No reduction needed, preserving all dims: {preserve_dims}")
        
        print(f"[DEBUG AlongTrackError] Final result:")
        print(f"  dims: {ate.dims}")
        print(f"  coords: {list(ate.coords)}")
        if "init_time" in ate.coords:
            print(f"  ✓ init_time PRESERVED in final result")
        else:
            print(f"  ⚠️ init_time NOT in final result")
        
        return ate


class CrossTrackError(BaseMetric):
    """Cross-track component of tropical cyclone position error.

    Projects the great‐circle displacement between forecast and observed
    positions onto the axis *perpendicular* to the observed track heading.

    Positive values indicate the forecast storm is to the **right** of the
    observed heading; negative values indicate it is to the **left**.  This
    metric isolates directional / steering‐flow biases.

    The output is indexed by ``lead_time`` (default ``preserve_dims``),
    averaging over all initializations for each lead time.

    Units default to **km**.
    """

    def __init__(
        self,
        name: str = "cross_track_error",
        preserve_dims: list = None,
        units: str = "km",
        **kwargs: Any,
    ):
        """Initialize CrossTrackError.

        Args:
            name: Metric name.  Defaults to ``"cross_track_error"``.
            preserve_dims: Dimension(s) to keep.  Defaults to ``["lead_time", "valid_time"]``
                to preserve init_time coordinate.
            units: ``"km"`` or ``"degrees"``.  Defaults to ``"km"``.
            **kwargs: Passed to ``BaseMetric.__init__``.
        """
        if preserve_dims is None:
            preserve_dims = ["lead_time", "valid_time"]
        super().__init__(name, preserve_dims=preserve_dims, **kwargs)
        self.units = units

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute cross-track error.

        Args:
            forecast: Forecast DataArray (TC track with lat/lon coords).
            target: Target DataArray (observed track with lat/lon coords).

        Returns:
            Cross-track error DataArray, reduced to ``preserve_dims``.
        """
        _, cte, _ = _compute_track_error_decomposition(
            forecast, target, units=self.units
        )
        preserve_dims = self.preserve_dims if isinstance(self.preserve_dims, list) else [self.preserve_dims]
        dims_to_reduce = [d for d in cte.dims if d not in preserve_dims]
        if dims_to_reduce:
            cte = cte.mean(dim=dims_to_reduce, skipna=True)
        return cte


class TotalTrackError(BaseMetric):
    """Total (unsigned) great‐circle position error for tropical cyclone tracks.

    Computes the haversine distance between forecast and observed positions at
    each matched valid_time.  This is the magnitude of the position error
    vector, without decomposition.

    The output is indexed by ``lead_time`` (default ``preserve_dims``),
    averaging over all initializations for each lead time.

    Units default to **km**.
    """

    def __init__(
        self,
        name: str = "total_track_error",
        preserve_dims: list = None,
        units: str = "km",
        **kwargs: Any,
    ):
        """Initialize TotalTrackError.

        Args:
            name: Metric name.  Defaults to ``"total_track_error"``.
            preserve_dims: Dimension(s) to keep.  Defaults to ``["lead_time", "valid_time"]``
                to preserve init_time coordinate.
            units: ``"km"`` or ``"degrees"``.  Defaults to ``"km"``.
            **kwargs: Passed to ``BaseMetric.__init__``.
        """
        if preserve_dims is None:
            preserve_dims = ["lead_time", "valid_time"]
        super().__init__(name, preserve_dims=preserve_dims, **kwargs)
        self.units = units

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute total track error.

        Args:
            forecast: Forecast DataArray (TC track with lat/lon coords).
            target: Target DataArray (observed track with lat/lon coords).

        Returns:
            Total track error DataArray, reduced to ``preserve_dims``.
        """
        _, _, tot = _compute_track_error_decomposition(
            forecast, target, units=self.units
        )
        preserve_dims = self.preserve_dims if isinstance(self.preserve_dims, list) else [self.preserve_dims]
        dims_to_reduce = [d for d in tot.dims if d not in preserve_dims]
        if dims_to_reduce:
            tot = tot.mean(dim=dims_to_reduce, skipna=True)
        return tot


class TrackIntensityMeanAbsoluteError(BaseMetric):
    """MAE of forecast vs observed TC intensity at each valid time.

    This uses the same underlying track intensity variable as
    ``LandfallIntensityMeanAbsoluteError`` (typically
    ``air_pressure_at_mean_sea_level``), but evaluates it at every matched
    valid time instead of only at landfall.
    """

    def __init__(
        self,
        name: str = "track_intensity_mae",
        preserve_dims: list = None,
        **kwargs: Any,
    ):
        if preserve_dims is None:
            preserve_dims = ["lead_time", "valid_time"]
        super().__init__(name, preserve_dims=preserve_dims, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute absolute MSLP intensity error along the full track."""
        # Match valid_time points between forecast and target, then compute
        # |forecast - observed| at each (lead_time, valid_time).
        fc_vts = pd.DatetimeIndex(forecast.valid_time.values)
        tgt_vts = pd.DatetimeIndex(target.valid_time.values)
        common_vts = fc_vts.intersection(tgt_vts)
        if len(common_vts) == 0:
            return utils._create_nan_dataarray(self.preserve_dims)

        tgt_vt_to_idx = {pd.Timestamp(vt): i for i, vt in enumerate(tgt_vts)}

        fc_vals = forecast.values.astype(float)
        tgt_vals = target.values.astype(float)
        out = np.full(forecast.shape, np.nan)

        for vt_idx, vt in enumerate(fc_vts):
            if vt not in common_vts:
                continue
            tgt_idx = tgt_vt_to_idx.get(vt)
            if tgt_idx is None:
                continue
            tgt_val = float(tgt_vals[tgt_idx])
            if np.isnan(tgt_val):
                continue

            for lt_idx in range(forecast.sizes["lead_time"]):
                fc_val = float(fc_vals[lt_idx, vt_idx])
                if np.isnan(fc_val):
                    continue
                out[lt_idx, vt_idx] = np.abs(fc_val - tgt_val)

        # Preserve all useful coords from forecast (including init_time if present).
        coords = {
            "lead_time": forecast.lead_time.values,
            "valid_time": forecast.valid_time.values,
        }
        for coord_name in forecast.coords:
            if coord_name not in coords:
                coords[coord_name] = forecast.coords[coord_name]

        result = xr.DataArray(
            out,
            dims=["lead_time", "valid_time"],
            coords=coords,
            name=self.name,
        )

        preserve_dims = (
            self.preserve_dims if isinstance(self.preserve_dims, list) else [self.preserve_dims]
        )
        dims_to_reduce = [d for d in result.dims if d not in preserve_dims]
        if dims_to_reduce:
            result = result.mean(dim=dims_to_reduce, skipna=True)
        return result


