import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from joblib import dump, load

class MLModelSelector:
    def __init__(self, data, target_column, xgb_params, lgbm_params, catboost_params, split_date=None, split_ratio=0.8):
        self.data = data.sort_index()  # Ensure the data is sorted by date
        self.target_column = target_column
        self.split_date = split_date
        self.split_ratio = split_ratio
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.catboost_params = catboost_params
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.models = {}
        self.initialize_models()

    def prepare_data(self):
        """
        Prepares the data by splitting it into training and testing sets based on a split date or split ratio.
        """
        if self.split_date:
            train = self.data.loc[:self.split_date]
            test = self.data.loc[self.split_date:]
        else:
            split_idx = int(len(self.data) * self.split_ratio)
            train = self.data.iloc[:split_idx]
            test = self.data.iloc[split_idx:]

        X_train = train.drop(columns=[self.target_column])
        y_train = train[self.target_column]
        X_test = test.drop(columns=[self.target_column])
        y_test = test[self.target_column]

        return X_train, X_test, y_train, y_test

    def initialize_models(self):
        """
        Initializes the individual models and stores them in a dictionary.
        """
        self.models['XGBoost'] = xgb.XGBRegressor(**self.xgb_params)
        self.models['LightGBM'] = lgb.LGBMRegressor(**self.lgbm_params)
        self.models['CatBoost'] = cb.CatBoostRegressor(**self.catboost_params)
        estimators = [(name, model) for name, model in self.models.items()]
        meta_model = LinearRegression()
        self.models['Ensemble'] = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=5)

    def fit(self):
        """
        Fit the models to the training data.
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"{name} training completed.")

    def predict(self, model_name, X_test):
        """
        Predict using a specified model.
        """
        model = self.models.get(model_name)
        if model:
            return model.predict(X_test)
        else:
            print(f"Model {model_name} not found.")
            return None

    def evaluate_and_predict_models(self):
        """
        Evaluate all models using the test data, return their errors in a table format,
        and compile their predictions alongside the actual values into a DataFrame.
        Additionally, calculate the standard deviation of the prediction errors.
        """
        results = {}
        predictions = {"Actual": self.y_test}
        for name, model in self.models.items():
            y_pred = self.predict(name, self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            nrmse = rmse / (self.y_test.max() - self.y_test.min())  # Normalized RMSE
            errors = self.y_test - y_pred
            std_dev = np.std(errors)  # Standard deviation of the prediction errors
            
            results[name] = {
                'MSE': mse,
                'NRMSE': nrmse,
                'MAE': mae,
                'StdDev': std_dev  # Adding standard deviation to the results
            }
            predictions[name] = y_pred
        
        # Convert the results dictionary to a DataFrame for easy display
        results_df = pd.DataFrame.from_dict(results, orient='index')
        
        # Convert the predictions dictionary to a DataFrame for easy display
        predictions_df = pd.DataFrame(predictions)
        
        return results_df, predictions_df


    def save_model(self, model_name, filename):
        """
        Save a specified model to a file.
        """
        model = self.models.get(model_name)
        if model:
            dump(model, filename)
            print(f"{model_name} model saved to {filename}")
        else:
            print(f"Model {model_name} not found. No model saved.")
