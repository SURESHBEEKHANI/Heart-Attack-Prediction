# Import necessary libraries and modules
import os  # For file and directory operations
import sys  # For system-specific parameters and functions
from src.exception import CustomException

from src.logger import logging  # Adjusted to absolute import

import pandas as pd  # For data manipulation and analysis

from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from dataclasses import dataclass  # For creating data classes

#Import components for data transformation and model training
from src.components.data_transformation import DataTransformation ,DataTransformationConfig # Data transformation class
  # Configuration for data transformation
from src.components.model_tranier import  ModelTrainer, ModelTrainerConfig # type: ignore # Configuration for model training
 # type: ignore # Model training class


# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        # Initialize ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            df = pd.read_csv('notebook\data\heart.csv')
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initate_data_ingestion()

    # Create an instance of DataTransformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data,test_data)

    #Create an instance of ModelTrainer
    model_trainer = ModelTrainer()
    metrics = model_trainer.initiate_model_training(train_arr, test_arr)

