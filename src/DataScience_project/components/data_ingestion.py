# Database ----> data --> train test split
#  using my sql database

# MySQL --->train test split

import os
import sys
from src.DataScience_project.exception import CustomException
from src.DataScience_project.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.DataScience_project.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact', 'train.csv')
    test_data_path:str=os.path.join('artifact', 'test.csv')
    raw_data_path:str=os.path.join('artifact', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            # reading the data from mysql 
            df = read_sql_data()
            logging.info("Reading completed from mysql database")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )
        except Exception as e:
            raise CustomException(e,sys)