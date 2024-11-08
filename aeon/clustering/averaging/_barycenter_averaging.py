__maintainer__ = []

from typing import Optional, Union

import numpy as np

from aeon.clustering.averaging._ba_petitjean import petitjean_barycenter_average
from aeon.clustering.averaging._ba_random_subset_ssg import (
    random_subset_ssg_barycenter_average,
)
from aeon.clustering.averaging._ba_random_subset_ssg_lr import (
    lr_random_subset_ssg_barycenter_average,
)
from aeon.clustering.averaging._ba_subgradient import subgradient_barycenter_average


def elastic_barycenter_average(
    X: np.ndarray,
    distance: str = "dtw",
    max_iters: int = 50,
    tol: float = 1e-5,
    init_barycenter: Union[np.ndarray, str] = "mean",
    method: str = "petitjean",
    initial_step_size: float = 0.05,
    final_step_size: float = 0.005,
    precomputed_medoids_pairwise_distance: Optional[np.ndarray] = None,
    verbose: bool = False,
    random_state: Optional[int] = None,
    ba_subset_size: float = 1.0,
    return_distances: bool = False,
    count_number_distance_calls: bool = False,
    previous_cost: Optional[float] = None,
    previous_distance_to_centre: Optional[np.ndarray] = None,
    use_all_first_subset_ba_iteration: bool = False,
    lr_func: str = "simple",
    decay_rate: float = 0.1,
    min_step_size: float = 0.005,
    **kwargs,
) -> np.ndarray:
    """Compute the barycenter average of time series using a elastic distance.

    This is a utility function that computes the barycenter average of a collection of
    time series instances. The barycenter algorithm used can be select using the method
    parameter. The following methods are available:
    - 'petitjean': This implements an adapted version of 'petitjean' (original) DBA
    algorithm [1]_.
    - 'subgradient': This implements a stochastic subgradient DBA
    algorithm [2]_.

    Petitjean is slower but guaranteed to find the optimal solution. Stochastic
    subgradient is much faster but not guaranteed to find the optimal solution.

    For large datasets it is recommended to use the 'subgradient' method. This will
    estimate the barycenter much faster than the 'petitjean' method. However, the
    'subgradient' method is not guaranteed to find the optimal solution. If
    computational time is not an issue, it is recommended to use the
    'petitjean' method.

    Parameters
    ----------
    X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
            (n_cases, n_timepoints)
        A collection of time series instances to take the average from.
    distance: str, default='dtw'
        String defining the distance to use for averaging. Distance to
        compute similarity between time series. A list of valid strings for metrics
        can be found in the documentation form
        :func:`aeon.distances.get_distance_function`.
    max_iters: int, default=30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    init_barycenter: np.ndarray or, default=None
        The initial barycenter to use for the minimisation. If a np.ndarray is provided
        it must be of shape ``(n_channels, n_timepoints)``. If a str is provided it must
        be one of the following: ['mean', 'medoids', 'random'].
        - 'mean': Uses mean of the time series instances.
        - 'medoids': Uses medoids of the time series instances.
        = 'random': Uses a random time series instance.
    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is suggested by [2]_.
    method: str, default='petitjean'
        The method to use for the barycenter averaging. Valid strings are:
        ['petitjean', 'subgradient'].
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                default=None
        Precomputed medoids pairwise.
    verbose: bool, default=False
        Boolean that controls the verbosity.
    random_state: int or None, default=None
        Random state to use for the barycenter averaging.
    ba_subset_size: float, default=1.0
        The proportion of the dataset to use for the barycenter averaging. If set to 1.0
        the full dataset will be used.
    return_distances: bool, default=False
        Boolean that controls if the distances to the average are returned.
    **kwargs
        Keyword arguments to pass to the distance metric.

    Returns
    -------
    np.ndarray of shape (n_channels, n_timepoints)
        Time series that is the average of the collection of instances provided.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    if X.shape[0] == 1:
        if return_distances:
            if count_number_distance_calls:
                return X[0], np.zeros(X.shape[0]), 0
            return X[0], np.zeros(X.shape[0])
        if count_number_distance_calls:
            return X[0], 0
        return X[0]
    if method == "petitjean":
        return petitjean_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            init_barycenter=init_barycenter,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            return_distances=return_distances,
            count_number_distance_calls=count_number_distance_calls,
            **kwargs,
        )
    elif method == "subgradient":
        return subgradient_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            init_barycenter=init_barycenter,
            initial_step_size=initial_step_size,
            final_step_size=final_step_size,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            return_distances=return_distances,
            count_number_distance_calls=count_number_distance_calls,
            **kwargs,
        )
    elif method == "random_subset_ssg":
        return random_subset_ssg_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            ba_subset_size=ba_subset_size,
            tol=tol,
            init_barycenter=init_barycenter,
            initial_step_size=initial_step_size,
            final_step_size=final_step_size,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            return_distances=return_distances,
            count_number_distance_calls=count_number_distance_calls,
            previous_cost=previous_cost,
            previous_distance_to_centre=previous_distance_to_centre,
            use_all_first_subset_ba_iteration=use_all_first_subset_ba_iteration,
            **kwargs,
        )
    elif method == "lr_random_subset_ssg":
        return lr_random_subset_ssg_barycenter_average(
            X,
            distance=distance,
            max_iters=max_iters,
            tol=tol,
            init_barycenter=init_barycenter,
            precomputed_medoids_pairwise_distance=precomputed_medoids_pairwise_distance,
            verbose=verbose,
            random_state=random_state,
            ba_subset_size=ba_subset_size,
            return_distances=return_distances,
            count_number_distance_calls=count_number_distance_calls,
            previous_cost=previous_cost,
            previous_distance_to_centre=previous_distance_to_centre,
            use_all_first_subset_ba_iteration=use_all_first_subset_ba_iteration,
            lr_func=lr_func,
            initial_step_size=initial_step_size,
            decay_rate=decay_rate,
            min_step_size=min_step_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Please use one of the following: "
            f"['petitjean', 'subgradient', 'random-subset-ssg']"
        )
