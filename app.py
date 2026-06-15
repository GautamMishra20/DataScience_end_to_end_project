from src.DataScience_project.logger import logging
from src.DataScience_project.exception import CustomException
import sys


if __name__=="__main__":
    logging.info("The execution has started")
    
    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)