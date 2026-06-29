import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple machine learning models using GridSearchCV.

    Args:
        X_train : Training features
        y_train : Training labels
        X_test : Testing features
        y_test : Testing labels
        models : Dictionary of models
        param : Dictionary of hyperparameters

    Returns:
        report : Dictionary containing model names and corresponding R2 scores.
    """

    try:

        report = {}

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

            # Update model with best parameters
            model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print("=" * 60)
            print(f"Model : {model_name}")
            print(f"Best Parameters : {gs.best_params_}")
            print(f"Train R2 Score : {train_model_score:.4f}")
            print(f"Test R2 Score : {test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)