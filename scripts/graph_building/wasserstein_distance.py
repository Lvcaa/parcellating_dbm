"""Reusable parcel-similarity distance computations.

This module deliberately has no knowledge of project paths, atlas files, or
output formats. Callers provide one one-dimensional value array per parcel and
decide whether and where to save the returned arrays.

Public API
----------
WassersteinDistanceComputer
    Configures and computes parcel similarities.
ParcelFeatures
    Holds the representations prepared from ordered parcel value arrays.
DistanceResult
    Holds the requested in-memory outputs from a computation.

The supported methods are ``"was"`` for approximate Wasserstein-1,
``"kl"`` for symmetrized Gaussian KL divergence, and ``"meiq"`` for a
median/IQR distance. Similarities are either ``exp(-distance)`` or
``1 / (1 + distance)``.

Examples
--------
Compute both the weighted-degree vector and dense similarity matrix::

    computer = WassersteinDistanceComputer(
        method="was", similarity_formula=SIM_FORMULA_EXP, quantile_count=15
    )
    features = computer.prepare_features([parcel_a_values, parcel_b_values])
    result = computer.compute(
        features,
        block_size=500,
        num_workers=4,
        compute_weighted_degree=True,
        return_similarity_matrix=True,
    )
    weighted_degree = result.weighted_degree
    similarity_matrix = result.similarity_matrix

For large parcellations, set ``return_similarity_matrix=False`` to avoid
allocating an N-by-N dense matrix while still computing weighted degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed


SIM_FORMULA_EXP = 1
SIM_FORMULA_INV1PW = 2
SIM_FORMULA_CHOICES = (SIM_FORMULA_EXP, SIM_FORMULA_INV1PW)
SIM_FORMULA_LABELS = {
    SIM_FORMULA_EXP: "exp(-W)",
    SIM_FORMULA_INV1PW: "1/(1+W)",
}

METHOD_WASSERSTEIN = "was"
METHOD_KL = "kl"
METHOD_MEDIAN_IQR = "meiq"
METHOD_CHOICES = (METHOD_WASSERSTEIN, METHOD_KL, METHOD_MEDIAN_IQR)
DEFAULT_QUANTILE_COUNT = 15


@dataclass(frozen=True)
class ParcelFeatures:
    """Representations of one ordered set of parcels.

    Each row corresponds to the same parcel in every field. ``quantiles`` is
    used for approximate Wasserstein-1; the remaining fields support the KL
    and median/IQR distances.

    Attributes
    ----------
    quantiles : numpy.ndarray
        Float array of shape ``(N, Q)``. ``N`` is the number of parcels and
        ``Q`` is the computer's ``quantile_count``.
    mean : numpy.ndarray
        Float array of shape ``(N,)`` containing each parcel's mean.
    variance : numpy.ndarray
        Positive float array of shape ``(N,)`` containing each parcel's
        variance after applying ``minimum_variance``.
    median : numpy.ndarray
        Float array of shape ``(N,)`` containing each parcel's median.
    iqr : numpy.ndarray
        Positive float array of shape ``(N,)`` containing each parcel's
        interquartile range after applying ``minimum_iqr``.

    Notes
    -----
    Create this class with :meth:`WassersteinDistanceComputer.prepare_features`
    in ordinary use. Supplying manually created features is supported when
    their shapes and numerical constraints match the attributes above.
    """

    quantiles: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    median: np.ndarray
    iqr: np.ndarray

    @property
    def parcel_count(self) -> int:
        return self.quantiles.shape[0]


@dataclass(frozen=True)
class DistanceResult:
    """In-memory results requested from :meth:`WassersteinDistanceComputer.compute`.

    Attributes
    ----------
    weighted_degree : numpy.ndarray or None
        Float64 vector of shape ``(N,)`` when ``compute_weighted_degree=True``;
        otherwise ``None``. Each entry is the mean of all off-diagonal
        similarities in the corresponding row.
    similarity_matrix : numpy.ndarray or None
        Float32 array of shape ``(N, N)`` when
        ``return_similarity_matrix=True``; otherwise ``None``. The diagonal is
        exactly one.
    """

    weighted_degree: np.ndarray | None
    similarity_matrix: np.ndarray | None


@dataclass(frozen=True)
class WassersteinDistanceComputer:
    """Compute parcel similarities from their value distributions.

    Parameters control computation only. The caller supplies parcel values,
    chooses which returned arrays to retain, and owns all persistence.

    Parameters
    ----------
    method : {"was", "kl", "meiq"}, default="was"
        Distance definition. ``"was"`` computes the mean absolute difference
        between corresponding quantiles. ``"kl"`` computes symmetrized KL
        divergence between Gaussian summaries. ``"meiq"`` divides the absolute
        median difference by the mean of the two IQRs.
    similarity_formula : {1, 2}, default=1
        Transform from non-negative distance to similarity. Use
        ``SIM_FORMULA_EXP`` (1) for ``exp(-distance)`` or
        ``SIM_FORMULA_INV1PW`` (2) for ``1 / (1 + distance)``.
    quantile_count : int, default=15
        Number of evenly spaced quantiles used by the ``"was"`` method. Every
        parcel is interpolated to this many values before distances are
        calculated. Must be at least one.
    minimum_variance : float, default=1e-6
        Lower bound applied to a parcel variance used by ``"kl"``. Must be
        positive.
    minimum_iqr : float, default=1e-6
        Lower bound applied to a parcel IQR used by ``"meiq"``. Must be
        positive.

    Notes
    -----
    The class retains configuration only; it does not retain subject data or
    write files. Create one instance for a chosen method and reuse it for each
    subject or parcellation. ``compute`` processes rows in blocks, so it can
    calculate weighted degree without retaining the dense matrix.
    """

    method: str = METHOD_WASSERSTEIN
    similarity_formula: int = SIM_FORMULA_EXP
    quantile_count: int = DEFAULT_QUANTILE_COUNT
    minimum_variance: float = 1e-6
    minimum_iqr: float = 1e-6

    def __post_init__(self) -> None:
        if self.method not in METHOD_CHOICES:
            raise ValueError(
                f"Unsupported method: {self.method!r}; choose from {METHOD_CHOICES}"
            )
        if self.similarity_formula not in SIM_FORMULA_CHOICES:
            raise ValueError(
                "Unsupported similarity formula: "
                f"{self.similarity_formula!r}; choose from {SIM_FORMULA_CHOICES}"
            )
        if self.quantile_count < 1:
            raise ValueError("quantile_count must be at least 1")
        if self.minimum_variance <= 0.0:
            raise ValueError("minimum_variance must be positive")
        if self.minimum_iqr <= 0.0:
            raise ValueError("minimum_iqr must be positive")

    def quantile_grid(self, values: np.ndarray) -> np.ndarray:
        """Return sorted values interpolated to this computer's quantile grid.

        Parameters
        ----------
        values : numpy.ndarray
            Finite one-dimensional array containing all values for one parcel.
            Its length may differ from ``quantile_count``.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(quantile_count,)``. If ``values`` already
            has that length, the sorted values are returned without
            interpolation.

        Raises
        ------
        ValueError
            If ``values`` is empty, not one-dimensional, or contains a
            non-finite value.
        """
        values = self._validate_parcel_values(values)
        sorted_values = np.sort(values.astype(np.float32, copy=False))
        value_count = len(sorted_values)
        if value_count == self.quantile_count:
            return sorted_values

        source_indices = np.arange(value_count, dtype=np.float32)
        source_quantiles = (source_indices + 0.5) / value_count
        target_indices = np.arange(self.quantile_count, dtype=np.float32)
        target_quantiles = (target_indices + 0.5) / self.quantile_count
        interpolated_values = np.interp(
            target_quantiles, source_quantiles, sorted_values
        )
        return interpolated_values.astype(np.float32)

    def prepare_features(self, parcel_values: Sequence[np.ndarray]) -> ParcelFeatures:
        """Build all distance representations for ordered parcel value arrays.

        Parameters
        ----------
        parcel_values : sequence of numpy.ndarray
            One finite, non-empty, one-dimensional array per parcel. Array
            lengths may differ between parcels. Sequence order defines the
            parcel order in every result.

        Returns
        -------
        ParcelFeatures
            Quantile, mean, variance, median, and IQR representations for the
            supplied parcels.

        Raises
        ------
        ValueError
            If no parcels are supplied or any parcel array is empty,
            multidimensional, or contains non-finite values.
        """
        if not parcel_values:
            raise ValueError("At least one parcel is required")

        quantile_rows: list[np.ndarray] = []
        means: list[float] = []
        variances: list[float] = []
        medians: list[float] = []
        iqrs: list[float] = []

        for values in parcel_values:
            values = self._validate_parcel_values(values)
            quantile_row = self.quantile_grid(values)
            quantile_rows.append(quantile_row)

            mean = float(np.mean(values, dtype=np.float64))
            means.append(mean)

            variance = float(np.var(values, dtype=np.float64))
            bounded_variance = max(variance, self.minimum_variance)
            variances.append(bounded_variance)

            first_quartile, third_quartile = np.percentile(values, [25, 75])
            median = float(np.median(values))
            medians.append(median)

            iqr = float(third_quartile - first_quartile)
            bounded_iqr = max(iqr, self.minimum_iqr)
            iqrs.append(bounded_iqr)

        quantile_matrix = np.stack(quantile_rows, axis=0)
        mean_vector = np.asarray(means, dtype=np.float64)
        variance_vector = np.asarray(variances, dtype=np.float64)
        median_vector = np.asarray(medians, dtype=np.float64)
        iqr_vector = np.asarray(iqrs, dtype=np.float64)
        return ParcelFeatures(
            quantiles=quantile_matrix,
            mean=mean_vector,
            variance=variance_vector,
            median=median_vector,
            iqr=iqr_vector,
        )

    def similarity_block(
        self, features: ParcelFeatures, start: int, stop: int
    ) -> np.ndarray:
        """Return similarities for rows ``[start, stop)`` against every parcel.

        Parameters
        ----------
        features : ParcelFeatures
            Features from :meth:`prepare_features`, or manually supplied
            features that match this computer's ``quantile_count``.
        start : int
            Inclusive index of the first requested parcel row.
        stop : int
            Exclusive index after the final requested parcel row.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(stop - start, N)``. Each value is in
            ``(0, 1]`` and the entries corresponding to the global diagonal
            are exactly one.

        Raises
        ------
        ValueError
            If the feature arrays are malformed, contain invalid values, or
            the requested interval is outside ``[0, N]``.
        """
        self._validate_features(features)
        parcel_count = features.parcel_count
        if not 0 <= start < stop <= parcel_count:
            raise ValueError(
                f"Invalid block [{start}, {stop}) for {parcel_count} parcels"
            )

        if self.method == METHOD_WASSERSTEIN:
            distances = self._wasserstein_distances(features.quantiles, start, stop)
        elif self.method == METHOD_KL:
            distances = self._kl_distances(
                features.mean, features.variance, start, stop
            )
        else:
            distances = self._median_iqr_distances(
                features.median, features.iqr, start, stop
            )

        similarities = self._distances_to_similarity(distances)
        for local_index, global_index in enumerate(range(start, stop)):
            similarities[local_index, global_index] = 1.0
        return similarities

    def compute(
        self,
        features: ParcelFeatures,
        *,
        block_size: int = 500,
        num_workers: int = 1,
        compute_weighted_degree: bool = True,
        return_similarity_matrix: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DistanceResult:
        """Compute requested in-memory results from prepared parcel features.

        Parameters
        ----------
        features : ParcelFeatures
            Features from :meth:`prepare_features`, or manually supplied
            features that match this computer's ``quantile_count``.
        block_size : int, default=500
            Number of parcel rows per similarity block. A block materializes an
            array of shape ``(block_size, N)`` before its result is consumed.
            Must be at least one.
        num_workers : int, default=1
            Number of Joblib process workers. Must be at least one.
        compute_weighted_degree : bool, default=True
            Return each parcel's mean off-diagonal similarity. The calculation
            excludes self-loops and divides by ``N - 1``.
        return_similarity_matrix : bool, default=False
            Return the full float32 matrix. Set this to ``False`` to avoid its
            ``N * N * 4`` byte allocation.
        progress_callback : callable or None, default=None
            Optional callable receiving ``(completed_blocks, total_blocks)``
            after each completed block. It can report progress without giving
            this computation module responsibility for console output.

        Returns
        -------
        DistanceResult
            Contains each requested output; an output not requested is ``None``.

        Raises
        ------
        ValueError
            If fewer than two parcels are present, an option is invalid, both
            outputs are disabled, or ``features`` is malformed.

        Notes
        -----
        No values are saved to disk. A caller that needs persistent outputs
        should write ``result.weighted_degree`` or ``result.similarity_matrix``
        after this method returns.
        """
        self._validate_features(features)
        parcel_count = features.parcel_count
        if parcel_count < 2:
            raise ValueError("At least two parcels are required")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if not compute_weighted_degree and not return_similarity_matrix:
            raise ValueError("Request weighted degree, similarity matrix, or both")

        blocks = [
            (start, min(start + block_size, parcel_count))
            for start in range(0, parcel_count, block_size)
        ]
        weighted_degree: np.ndarray | None = None
        if compute_weighted_degree:
            weighted_degree = np.empty(parcel_count, dtype=np.float64)

        similarity_matrix: np.ndarray | None = None
        if return_similarity_matrix:
            similarity_matrix = np.empty(
                (parcel_count, parcel_count), dtype=np.float32
            )

        block_results = Parallel(
            n_jobs=num_workers, prefer="processes", return_as="generator"
        )(
            delayed(self._compute_block)(features, start, stop)
            for start, stop in blocks
        )
        normalization_count = parcel_count - 1
        total_blocks = len(blocks)
        for completed_blocks, (start, stop, similarities) in enumerate(
            block_results, start=1
        ):
            if weighted_degree is not None:
                similarity_sum = similarities.sum(axis=1, dtype=np.float64)
                off_diagonal_sum = similarity_sum - 1.0
                block_weighted_degree = off_diagonal_sum / normalization_count
                weighted_degree[start:stop] = block_weighted_degree
            if similarity_matrix is not None:
                similarity_matrix[start:stop] = similarities
            if progress_callback is not None:
                progress_callback(completed_blocks, total_blocks)

        return DistanceResult(
            weighted_degree=weighted_degree, similarity_matrix=similarity_matrix
        )

    def _compute_block(
        self, features: ParcelFeatures, start: int, stop: int
    ) -> tuple[int, int, np.ndarray]:
        return start, stop, self.similarity_block(features, start, stop)

    @staticmethod
    def _validate_parcel_values(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Each parcel must contain a non-empty one-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("Parcel values must all be finite")
        return values

    def _validate_features(self, features: ParcelFeatures) -> None:
        parcel_count = features.parcel_count
        quantile_matrix = features.quantiles
        quantile_shape_is_valid = (
            quantile_matrix.ndim == 2
            and quantile_matrix.shape[1] == self.quantile_count
        )
        if not quantile_shape_is_valid:
            raise ValueError(
                "quantiles must have shape (parcel_count, quantile_count) for this computer"
            )

        self._validate_feature_vector("mean", features.mean, parcel_count)
        self._validate_feature_vector("variance", features.variance, parcel_count)
        self._validate_feature_vector("median", features.median, parcel_count)
        self._validate_feature_vector("iqr", features.iqr, parcel_count)

        has_non_positive_variance = np.any(features.variance <= 0.0)
        has_non_positive_iqr = np.any(features.iqr <= 0.0)
        if has_non_positive_variance or has_non_positive_iqr:
            raise ValueError("variance and iqr must be positive")

    @staticmethod
    def _validate_feature_vector(
        name: str, values: np.ndarray, parcel_count: int
    ) -> None:
        has_one_value_per_parcel = values.ndim == 1 and len(values) == parcel_count
        if not has_one_value_per_parcel:
            raise ValueError(f"{name} must have one value per parcel")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _wasserstein_distances(
        quantiles: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        block_quantiles = quantiles[start:stop]
        parcel_count = len(quantiles)
        quantile_count = quantiles.shape[1]
        distances = np.zeros((len(block_quantiles), parcel_count), dtype=np.float32)
        for quantile_index in range(quantile_count):
            block_values = block_quantiles[:, quantile_index : quantile_index + 1]
            reference_values = quantiles[:, quantile_index]
            absolute_differences = np.abs(block_values - reference_values)
            distances += absolute_differences
        return distances / quantile_count

    @staticmethod
    def _kl_distances(
        mean: np.ndarray, variance: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        block_means = mean[start:stop, None]
        reference_means = mean[None, :]
        block_variances = variance[start:stop, None]
        reference_variances = variance[None, :]

        squared_mean_difference = (block_means - reference_means) ** 2
        kl_block_to_reference = 0.5 * np.log(
            reference_variances / block_variances
        )
        kl_block_to_reference += (
            block_variances + squared_mean_difference
        ) / (2 * reference_variances) - 0.5

        kl_reference_to_block = 0.5 * np.log(
            block_variances / reference_variances
        )
        kl_reference_to_block += (
            reference_variances + squared_mean_difference
        ) / (2 * block_variances) - 0.5

        symmetric_kl = 0.5 * (kl_block_to_reference + kl_reference_to_block)
        non_negative_kl = np.clip(symmetric_kl, 0.0, None)
        return non_negative_kl.astype(np.float32)

    @staticmethod
    def _median_iqr_distances(
        median: np.ndarray, iqr: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        block_medians = median[start:stop, None]
        reference_medians = median[None, :]
        median_difference = np.abs(block_medians - reference_medians)

        block_iqrs = iqr[start:stop, None]
        reference_iqrs = iqr[None, :]
        mean_iqr = 0.5 * (block_iqrs + reference_iqrs)

        distances = median_difference / mean_iqr
        return distances.astype(np.float32)

    def _distances_to_similarity(self, distances: np.ndarray) -> np.ndarray:
        if self.similarity_formula == SIM_FORMULA_EXP:
            similarities = np.exp(-distances).astype(np.float32, copy=False)
        else:
            similarities = 1.0 / (1.0 + distances)

        has_non_finite_values = not np.isfinite(similarities).all()
        has_non_positive_values = np.any(similarities <= 0.0)
        has_values_above_one = np.any(similarities > 1.0)
        if has_non_finite_values or has_non_positive_values or has_values_above_one:
            raise ValueError("Distance calculation produced invalid similarities")
        return similarities
