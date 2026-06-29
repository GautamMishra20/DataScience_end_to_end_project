import os
import sys
import warnings
import dill
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from urllib.parse import urlparse

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    
    try:
        
        report = {}

        skops_trusted = [
            "sklearn.metrics._dist_metrics.ManhattanDistance64",
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._ball_tree.BallTree",
            "sklearn.neighbors._kd_tree.KDTree",
        ]

        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme

        for model_name, model in models.items():

            para = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )
            
            gs.fit(X_train, y_train)

            
            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)

            
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            train_r2  = r2_score(y_train, y_train_pred)
            test_r2   = r2_score(y_test,  y_test_pred)
            test_mae  = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[model_name] = test_r2

            with mlflow.start_run(run_name=model_name, nested=True):

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(gs.best_params_)

                mlflow.log_metric("train_r2",  train_r2)
                mlflow.log_metric("test_r2",   test_r2)
                mlflow.log_metric("test_mae",  test_mae)
                mlflow.log_metric("test_rmse", test_rmse)

                # Use native flavors for XGBoost and CatBoost
                if model_name == "XGBoost":
                    mlflow.xgboost.log_model(
                        model, "model",
                        registered_model_name=model_name
                    )
                elif model_name == "CatBoost":
                    mlflow.catboost.log_model(
                        model, "model",
                        registered_model_name=model_name
                    )
                elif tracking_url_type != "file":
                    mlflow.sklearn.log_model(
                        model, "model",
                        registered_model_name=model_name,
                        skops_trusted_types=skops_trusted
                    )
                else:
                    mlflow.sklearn.log_model(
                        model, "model",
                        skops_trusted_types=skops_trusted
                    )

            print("=" * 60)
            print(f"Model      : {model_name}")
            print(f"Best Params: {gs.best_params_}")
            print(f"Train R²   : {train_r2:.4f}")
            print(f"Test R²    : {test_r2:.4f}  |  MAE: {test_mae:.4f}  |  RMSE: {test_rmse:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)