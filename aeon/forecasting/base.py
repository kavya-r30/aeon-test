"""BaseForecaster class.

A simplified first base class for foreacasting models. The focus here is on a
specific form of forecasting: longer series, long winodws and single step forecasting.

aeon enhancement proposal
https://github.com/aeon-toolkit/aeon-admin/pull/14

"""

from abc import abstractmethod

from aeon.base import BaseSeriesEstimator


class BaseForecaster(BaseSeriesEstimator):
    """
    Abstract base class for time series forecasters.

    The base forecaster specifies the methods and method signatures that all
    forecasters have to implement. Attributes with an underscore suffix are set in the
    method fit.

    Parameters
    ----------
    horizon : int, default =1
        The number of time steps ahead to forecast. If horizon is one, the forecaster
        will learn to predict one point ahead
    window : int or None
        The window prior to the current time point to use in forecasting. So if
        horizon is one, forecaster will train using points $i$ to $window+i-1$ to
        predict value $window+i$. If horizon is 4, forecaster will used points $i$
        to $window+i-1$ to predict value $window+i+3$. If None, the algorithm will
        internally determine what data to use to predict `horizon` steps ahead.
    """

    # TODO: add any forecasting specific tags
    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "X_inner_type": "np.ndarray",
    }

    def __init__(self, horizon=1, window=None, axis=1):
        self.horizon = horizon
        self.window = window
        self._is_fitted = False
        super().__init__(axis)

    def fit(self, y, exog=None):
        """Fit forecaster to series y.

        Fit a forecaster to predict self.horizon steps ahead using y.

        Parameters
        ----------
        y : np.ndarray
            A time series on which to learn a forecaster to predict horizon ahead
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        self
            Fitted BaseForecaster.
        """
        # Validate y

        # Convert if necessary

        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        # Validate exog
        self._is_fitted = True
        return self._fit(y, exog)

    @abstractmethod
    def _fit(self, y, exog=None): ...

    def predict(self, y=None, exog=None):
        """Predict the next horizon steps ahead.

        Parameters
        ----------
        y : np.ndarray, default = None
            A time series to predict the next horizon value for. If None,
            predict the next horizon value after series seen in fit.
        exog : np.ndarray, default =None
            Optional exogenous time series data assumed to be aligned with y

        Returns
        -------
        float
            single prediction self.horizon steps ahead of y.
        """
        if not self._is_fitted:
            raise ValueError("Forecaster must be fitted before predicting")
        if exog is not None:
            raise NotImplementedError("Exogenous variables not yet supported")
        # Validate exog
        self._is_fitted = True
        return self._predict(y, exog)

    @abstractmethod
    def _predict(self, y=None, exog=None): ...

    def forecast(self, y, X=None):
        """

        Forecast basically fit_predict.

        Returns
        -------
        np.ndarray
            single prediction directly after the last point in X.
        """
        return self._forecast(y, X)

    @abstractmethod
    def _forecast(self, y=None, exog=None): ...
