import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingRegressor

class MLProcessor:
    def __init__(self):
        # Initialize models
        self.xgb = XGBRegressor()
        self.lgbm = LGBMRegressor()
        self.catboost = CatBoostRegressor(silent=True)
        self.ensemble = None  # Will be defined after fitting models

    def fit(self, X_train, y_train):
        """
        Fit the models to the training data.

        Parameters:
        - X_train: DataFrame containing the training features.
        - y_train: Series containing the training target.
        """
        # Fit individual models
        self.xgb.fit(X_train, y_train)
        self.lgbm.fit(X_train, y_train)
        self.catboost.fit(X_train, y_train)

        # Create an ensemble model
        estimators = [('xgb', self.xgb), ('lgbm', self.lgbm), ('catboost', self.catboost)]
        self.ensemble = VotingRegressor(estimators=estimators)
        self.ensemble.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions with the ensemble model on the test data.

        Parameters:
        - X_test: DataFrame containing the test features.

        Returns:
        - np.array with predictions.
        """
        if self.ensemble is not None:
            return self.ensemble.predict(X_test)
        else:
            raise Exception("Models are not fitted yet. Call the fit method first.")

    def evaluate(self, X_test, y_true):
        """
        Evaluate the ensemble model performance.

        Parameters:
        - X_test: DataFrame containing the test features.
        - y_true: Series or array containing the true values of the target variable.

        Returns:
        - Dictionary with RMSE and MAE metrics.
        """
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        return {"RMSE": rmse, "MAE": mae}
