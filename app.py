from src.DataScience_project.logger import logging
from src.DataScience_project.exception import CustomException
import sys
from src.DataScience_project.components.data_ingestion import DataIngestion
from src.DataScience_project.components.data_ingestion import DataIngestionConfig



if __name__=="__main__":
    logging.info("The execution has started")
    
    try:
        # data_ingestion_config = DataIngestionConfig()
       data_ingestion = DataIngestion()
       data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)