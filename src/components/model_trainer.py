import os
import sys
import numpy as np

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting the data into train and test part!!")
            
            X_train, y_train,X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False,allow_writing_files=False)
            }
            
            params = {

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0],
                    "max_depth": [3, 5]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 1.0],
                    "n_estimators": [50, 100, 200]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200]
                }

            }
            
            logging.info("Evaluating all models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found.", sys)

            logging.info(f"Best Model Found: {best_model_name}")
            logging.info(f"Best Model Test R2 Score: {best_model_score}")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score: {r2_square}")

            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)