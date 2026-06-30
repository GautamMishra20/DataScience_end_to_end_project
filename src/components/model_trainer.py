import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.components.model_explainer import ModelExplainer

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
            
            db_path = os.path.join(os.getcwd(), "mlflow.db").replace("\\", "/")
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            
            mlflow.set_experiment("Student Performance Prediction")

            with mlflow.start_run(run_name="Full Training Session"):

                logging.info("Evaluating all models")
                model_report = evaluate_models(
                    X_train=X_train, y_train=y_train,
                    X_test=X_test,   y_test=y_test,
                    models=models,   param=params,
                )

                best_model_score = max(model_report.values())
                best_model_name  = max(model_report, key=model_report.get)
                best_model       = models[best_model_name]

                if best_model_score < 0.60:
                    raise CustomException("No best model found.", sys)

                logging.info(f"Best Model: {best_model_name} | R²: {best_model_score:.4f}")

                # Log best model summary to parent run
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_test_r2", best_model_score)

                # Final metrics on best model
                predicted = best_model.predict(X_test)
                final_r2   = r2_score(y_test, predicted)
                final_mae  = mean_absolute_error(y_test, predicted)
                final_rmse = np.sqrt(mean_squared_error(y_test, predicted))

                mlflow.log_metric("final_r2",   final_r2)
                mlflow.log_metric("final_mae",  final_mae)
                mlflow.log_metric("final_rmse", final_rmse)

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                explainer_obj = ModelExplainer()
                explainer_obj.generate_shap_plots(
                    test_df_path=os.path.join('artifact', 'test.csv')
                )

                shap_dir = os.path.join('artifact', 'shap_plots')
                for plot_file in os.listdir(shap_dir):
                    mlflow.log_artifact(os.path.join(shap_dir,plot_file), artifact_path="shap_plots")
            
            logging.info(f"Final r2 Score: {final_r2:.4f}")
            return final_r2
            
        except Exception as e:
            raise CustomException(e,sys)