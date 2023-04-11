import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass  #used to create class variables

@dataclass                         #used @dataclass here since we are only defining variables, in this case, paths where the trai, test and raw data will be saved
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")    # artifacts is a folder that will be created further down below
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "raw.csv")

class DataIngestion:            #used __init__ constructor here since we're not defining variables; we have other functions inside the class
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):    #this function reads the data from where it is stored
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as a dataframe")

            # create the folder for the train, test and raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ =="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()