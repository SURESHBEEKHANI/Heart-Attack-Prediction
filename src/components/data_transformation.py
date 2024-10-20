import os  # For file and directory operations
import sys  # For system-specific parameters and functions
import numpy as np  # For handling arrays
import pandas as pd  # For data manipulation and analysis

from sklearn.compose import ColumnTransformer  # For applying multiple preprocessing steps
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.pipeline import Pipeline  # For constructing machine learning pipelines
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # For encoding and scaling

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # For logging information
from src.utils import save_object  # For saving objects like preprocessing pipelines


# Configuration for Data Transformation, including preprocessor file path
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data Transformation Class
class DataTransformation:
    def __init__(self):
        # Ensure the DataTransformationConfig is initialized correctly
        logging.info("Initializing DataTransformationConfig")
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for creating and returning a preprocessing object 
        for transforming the data (both numerical and categorical features).
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

            # Define the custom ranking for each ordinal variable
            sex_categories = ['M', 'F']
            chestPainType_categories = ['ATA', 'NAP', 'ASY', 'TA']
            RestingECG_categories = ['Normal', 'ST', 'LVH']
            ExerciseAngina_categories = ['N', 'Y']
            ST_Slope_categories = ['Up', 'Flat', 'Down']

            # Numerical Pipeline: Impute missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with the median
                    ('scaler', StandardScaler())  # Standardize numerical features
                ]
            )

            # Categorical Pipeline: Impute missing values, apply Ordinal encoding, and scale
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent
                    ('ordinal_encoder', OrdinalEncoder(categories=[
                        sex_categories,
                        chestPainType_categories,
                        RestingECG_categories,
                        ExerciseAngina_categories,
                        ST_Slope_categories
                    ])),  # Ordinal encoding for categorical features
                    ('scaler', StandardScaler())  # Standardize categorical features after encoding
                ]
            )

            logging.info(f'Categorical Columns: {categorical_cols}')
            logging.info(f'Numerical Columns: {numerical_cols}')

            # Combine the numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info('Exception occurred in Data Transformation Phase')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function reads the train and test data, applies preprocessing, and 
        returns the transformed datasets and the preprocessing object.
        '''
        try:
            # Reading train and test data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Train and test data successfully read')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # Define target column and columns to drop
            target_column_name = 'HeartDisease'
            drop_columns = [target_column_name]

            # Splitting features and target for training and testing datasets
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying the preprocessing object to training and testing data
            logging.info('Applying preprocessing object on training and testing datasets')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenating the transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved successfully')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation function')
            raise CustomException(e, sys)
