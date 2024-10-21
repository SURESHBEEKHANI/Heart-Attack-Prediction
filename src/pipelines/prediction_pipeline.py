import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass  # Initialize the PredictPipeline class

    def predict(self, features):
        """
        Predict the target variable using the pre-trained model.

        Parameters:
            features (DataFrame): DataFrame containing input features for prediction.

        Returns:
            pred: Prediction results from the model.
        """
        try:
            # Load preprocessor and model from the specified paths
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Scale the input features using the loaded preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the scaled data
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            # Log the exception and raise a custom exception
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 age: int,
                 sex: str,
                 chest_pain_type: str,
                 resting_bp: float,
                 cholesterol: float,
                 fasting_bs: int,
                 resting_ecg: str,  # Added resting ECG
                 max_hr: float,
                 exercise_angina: str,
                 oldpeak: float,
                 st_slope: str):
        """
        Initialize custom data for prediction.

        Parameters:
            age (float): Age of the patient.
            sex (str): Gender of the patient (M/F).
            chest_pain_type (str): Type of chest pain (ATA/NAP/ASY).
            resting_bp (float): Resting blood pressure.
            cholesterol (float): Cholesterol level.
            fasting_bs (int): Fasting blood sugar level (0 or 1).
            resting_ecg (str): Resting ECG results (Normal/ST).
            max_hr (float): Maximum heart rate achieved.
            oldpeak (float): ST depression induced by exercise relative to rest.
            exercise_angina (str): Whether the patient experiences angina during exercise (Y/N).
            st_slope (str): The slope of the ST segment (Up/Flat/Down).
        """
        if age is None or resting_bp is None or cholesterol is None or max_hr is None or oldpeak is None:
            raise ValueError("Numeric fields cannot be None")

        self.age = age
        self.sex = sex
        self.chest_pain_type = chest_pain_type
        self.resting_bp = resting_bp
        self.cholesterol = cholesterol
        self.fasting_bs = fasting_bs
        self.resting_ecg = resting_ecg
        self.max_hr = max_hr
        self.oldpeak = oldpeak
        self.exercise_angina = exercise_angina
        self.st_slope = st_slope

    def get_data_as_dataframe(self):
        """
        Convert the input data into a pandas DataFrame.

        Returns:
            DataFrame: DataFrame containing the input features.
        """
        try:
            # Create a dictionary with the input data
            custom_data_input_dict = {
                'Age': [self.age],
                'Sex': [self.sex],
                'ChestPainType': [self.chest_pain_type],
                'RestingBP': [self.resting_bp],
                'Cholesterol': [self.cholesterol],
                'FastingBS': [self.fasting_bs],
                'RestingECG': [self.resting_ecg],
                'MaxHR': [self.max_hr],
                'Oldpeak': [self.oldpeak],
                'ExerciseAngina': [self.exercise_angina],
                'ST_Slope': [self.st_slope]
            }
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered successfully')
            logging.info(f"DataFrame contents: {df}")  # Log the DataFrame contents
            return df
        except Exception as e:
            # Log any exceptions that occur during DataFrame creation
            logging.info('Exception occurred in getting dataframe')
            raise CustomException(e, sys)
