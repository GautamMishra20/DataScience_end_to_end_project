import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    logging.info("Training Pipeline Started")

    try:

        # Data Ingestion
        data_ingestion = DataIngestion()

        train_data_path, test_data_path = (
            data_ingestion.initiate_data_ingestion()
        )

        logging.info("Data Ingestion Completed")

        # Data Transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
        )

        logging.info("Data Transformation Completed")

        # Model Training
        model_trainer = ModelTrainer()

        r2_score = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        print("=" * 60)
        print(f"Training Completed Successfully")
        print(f"Best Model Test R² Score : {r2_score:.4f}")
        print("=" * 60)

        logging.info(f"Training Completed. R2 Score : {r2_score}")

    except Exception as e:
        raise CustomException(e, sys)