""".

Example code

from aeon.datasets import load_classification
from sklearn.linear_model import RidgeClassifierCV
from aeon.classification.shapelet_based import RDSTClassifier

X_train, y_train = load_classification("ArrowHead", split="train")
X_test, y_test = load_classification("ArrowHead", split="test")
a = np.abs(np.diff(X_train).flatten())
g = GridIndexShapeletTransform(
    [a.mean() + a.std() * 7],
    3,
    min_size_bucket=3,
    n_jobs=-1,
    random_state=42,
).fit(X_train, y_train)

T_train = g.transform(X_train)
T_test = g.transform(X_test)
print(T_train.shape)
rdg = RidgeClassifierCV().fit(T_train, y_train)
print(rdg.score(T_test, y_test))
print(RDSTClassifier(n_jobs=-1).fit(X_train, y_train).score(X_test, y_test))

"""

import numpy as np
from numba import njit, prange
from numba.typed import List

from aeon.similarity_search.collection import GridIndexANN
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.general import (
    combinations_1d,
    get_all_subsequences,
    get_subsequence,
    normalise_subsequences,
    sliding_mean_std_one_series,
    z_normalise_series_2d,
)


@njit(fastmath=True, cache=True)
def compute_shapelet_dist_vector(
    X_subs: np.ndarray,
    values: np.ndarray,
):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    distance: CPUDispatcher
        A Numba function used to compute the distance between two multidimensional
        time series of shape (n_channels, length).

    Returns
    -------
    dist_vector : array, shape = (n_timestamps-(length-1)*dilation)
        The distance vector between the shapelets and candidate subsequences
    """
    n_subsequences, n_channels, length = X_subs.shape
    dist_vector = np.zeros(n_subsequences)
    for i_sub in prange(n_subsequences):
        for k in prange(n_channels):
            for i_len in prange(length):
                dist_vector[i_sub] += abs(X_subs[i_sub, k, i_len] - values[k, i_len])
    return dist_vector


@njit(fastmath=True, cache=True, parallel=True)
def dilated_shapelet_transform(
    X: np.ndarray,
    values,
    lengths,
    threshold,
    normalise,
    dilations,
):
    """Perform the shapelet transform with a set of shapelets and a set of time series.

    Parameters
    ----------
    X : array, shape (n_cases, n_channels, n_timepoints)
        Time series dataset
    shapelets : tuple
        The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (n_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - startpoints : array, shape (max_shapelets)
            Start points parameter of the shapelets
        - lengths : array, shape (n_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (n_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (n_shapelets)
            Threshold parameter of the shapelets
        - normalise : array, shape (n_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (n_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (n_shapelets, n_channels)
            Standard deviation of the shapelets
        - classes : array, shape (max_shapelets)
        An initialized (empty) startpoint array for each shapelet


    Returns
    -------
    X_new : array, shape=(n_cases, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        features from the distance vector computed on each time series.

    """
    n_shapelets = len(lengths)
    n_cases = len(X)
    n_ft = 3

    # (u_l * u_d , 2)
    params_shp = combinations_1d(lengths, dilations)

    X_new = np.zeros((n_cases, n_ft * n_shapelets))
    for i_params in prange(params_shp.shape[0]):
        length = params_shp[i_params, 0]
        dilation = params_shp[i_params, 1]
        id_shps = np.where((lengths == length) & (dilations == dilation))[0]

        for i_x in prange(n_cases):
            X_subs = get_all_subsequences(X[i_x], length, dilation)
            idx_no_norm = id_shps[np.where(~normalise[id_shps])[0]]
            for i_shp in idx_no_norm:
                X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                    compute_shapelet_features(X_subs, values[i_shp], threshold[i_shp])
                )

            idx_norm = id_shps[np.where(normalise[id_shps])[0]]
            if len(idx_norm) > 0:
                X_means, X_stds = sliding_mean_std_one_series(X[i_x], length, dilation)
                X_subs = normalise_subsequences(X_subs, X_means, X_stds)
                for i_shp in idx_norm:
                    X_new[i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)] = (
                        compute_shapelet_features(
                            X_subs, values[i_shp], threshold[i_shp]
                        )
                    )
    return X_new


@njit(fastmath=True, cache=True)
def compute_shapelet_features(
    X_subs: np.ndarray,
    values: np.ndarray,
    threshold: float,
):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    threshold : float
        The threshold parameter of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    _min = np.inf
    _argmin = np.inf
    _SO = 0

    n_subsequences, n_channels, length = X_subs.shape

    for i_sub in range(n_subsequences):
        _dist = 0
        for k in range(n_channels):
            for i_len in range(length):
                _dist += abs(X_subs[i_sub, k, i_len] - values[k, i_len])
        if _dist < _min:
            _min = _dist
            _argmin = i_sub
        if _dist < threshold:
            _SO += 1

    return np.float64(_min), np.float64(_argmin), np.float64(_SO)


class GridIndexShapeletTransform(BaseCollectionTransformer):
    """GridIndexShapeletTransform estimator."""

    _tags = {
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["numpy3D", "np-list"],
    }

    def __init__(
        self,
        grid_deltas,
        K,
        L_max=0.33,
        L_min=5,
        L_step=0.05,
        random_state=None,
        min_size_bucket=10,
        n_jobs=1,
    ):
        self.grid_deltas = np.array(grid_deltas, dtype=float)
        self.K = K
        self.L_min = L_min
        self.L_max = L_max
        self.L_step = L_step
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_size_bucket = min_size_bucket
        super().__init__()

    def _fit(self, X, y=None):
        self.index_norm_ = GridIndexANN(
            self.grid_deltas,
            self.K,
            L_max=self.L_max,
            L_min=self.L_min,
            random_state=self.random_state,
            normalize=True,
            n_jobs=self.n_jobs,
        ).fit(X)

        self.index_ = GridIndexANN(
            self.grid_deltas,
            self.K,
            L_max=self.L_max,
            L_min=self.L_min,
            random_state=self.random_state,
            normalize=False,
            n_jobs=self.n_jobs,
        ).fit(X)

        # Non normalized index
        _to_drop = []
        for key, bucket in self.index_.index_.items():
            if len(np.unique(bucket[:, 0])) < self.min_size_bucket:
                _to_drop.append(key)
            if y is not None:
                pass
                # Then by feature if y is specified ?
                # But then must be adaptable to clsf and regression
        for key in _to_drop:
            self.index_.index_.pop(key)

        # Normalized index
        _to_drop = []
        for key, bucket in self.index_norm_.index_.items():
            if len(np.unique(bucket[:, 0])) < self.min_size_bucket:
                _to_drop.append(key)
            if y is not None:
                pass
                # Then by feature if y is specified ?
                # But then must be adaptable to clsf and regression
        for key in _to_drop:
            self.index_norm_.index_.pop(key)

        shapelet_samples = np.asarray(
            [
                bucket[np.random.choice(range(len(bucket)))]
                for bucket in self.index_.index_.values()
            ],
            dtype=int,
        )
        shapelet_samples_norm = np.asarray(
            [
                bucket[np.random.choice(range(len(bucket)))]
                for bucket in self.index_norm_.index_.values()
            ],
            dtype=int,
        )
        n_shp = len(shapelet_samples) + len(shapelet_samples_norm)
        self.lengths_ = np.zeros(n_shp, dtype=int)
        self.thresholds_ = np.zeros(n_shp, dtype=int)
        self.dilations_ = np.zeros(n_shp, dtype=int)
        self.normalize_ = np.zeros(n_shp, dtype=bool)
        self.values_ = List()
        for i in range(len(shapelet_samples)):
            i_x = shapelet_samples[i, 0]
            j_x = shapelet_samples[i, 1]
            length = shapelet_samples[i, 2]
            d = shapelet_samples[i, 3]
            val = get_subsequence(X[i_x], j_x, length, d)
            self.normalize_[i] = False
            self.values_.append(val)
            self.lengths_[i] = length
            self.dilations_[i] = d

            i_same = np.random.choice(np.where(y == y[i_x])[0])
            subs = get_all_subsequences(X[i_same], length, d)
            self.thresholds_[i] = np.percentile(
                compute_shapelet_dist_vector(subs, val), np.random.uniform(5, 10)
            )

        for _i in range(len(shapelet_samples_norm)):
            i = _i + len(shapelet_samples)
            i_x = shapelet_samples_norm[_i, 0]
            j_x = shapelet_samples_norm[_i, 1]
            length = shapelet_samples_norm[_i, 2]
            d = shapelet_samples_norm[_i, 3]
            val = z_normalise_series_2d(get_subsequence(X[i_x], j_x, length, d))
            self.normalize_[i] = True
            self.values_.append(val)
            self.lengths_[i] = length
            self.dilations_[i] = d
            i_same = np.random.choice(np.where(y == y[i_x])[0])
            subs = get_all_subsequences(X[i_same], length, d)
            X_means, X_stds = sliding_mean_std_one_series(X[i_same], length, d)
            subs = normalise_subsequences(subs, X_means, X_stds)
            self.thresholds_[i] = np.percentile(
                compute_shapelet_dist_vector(subs, val), np.random.uniform(5, 10)
            )

    def _transform(
        self,
        X,
        y=None,
    ):
        return dilated_shapelet_transform(
            X,
            self.values_,
            self.lengths_,
            self.thresholds_,
            self.normalize_,
            self.dilations_,
        )
