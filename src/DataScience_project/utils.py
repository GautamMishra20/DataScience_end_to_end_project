# reading the data from mysql
import os
import sys
from src.DataScience_project.exception import CustomException
from src.DataScience_project.logger import logging

import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv('host')
user = os.getenv('users')
password = os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    logging.info('Reading SQL database started!!')
    try:
        mydb=pymysql.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )
        logging.info("Connection Established %s",mydb)
        df = pd.read_sql_query('select * from students', mydb)
        print(df.head())
        
        return df
        
    except Exception as e:
        raise CustomException(e,sys)