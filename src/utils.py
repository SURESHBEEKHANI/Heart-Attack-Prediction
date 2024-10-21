import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object) -> None:
    """Saves an object to a specified file path using dill.

    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(xtrain: np.ndarray, ytrain: np.ndarray, xtest: np.ndarray, ytest: np.ndarray, models: dict) -> dict:
    """Evaluates multiple models and returns their test scores.

    Args:
        xtrain (np.ndarray): Training features.
        ytrain (np.ndarray): Training labels.
        xtest (np.ndarray): Testing features.
        ytest (np.ndarray): Testing labels.
        models (dict): A dictionary of models to evaluate.

    Returns:
        dict: A dictionary containing model names and their corresponding test scores.

    Raises:
        CustomException: If an error occurs during evaluation.
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(xtrain, ytrain)

            # Predict Training data
            y_train_pred = model.predict(xtrain)

            # Predict Testing data
            y_test_pred = model.predict(xtest)

            # Get R2 scores for train and test data
            test_model_score = r2_score(ytest, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys)

def model_metrics(true: np.ndarray, predicted: np.ndarray) -> tuple:
    """Calculates regression metrics: MAE, RMSE, and R2 Score.

    Args:
        true (np.ndarray): True values.
        predicted (np.ndarray): Predicted values.

    Returns:
        tuple: A tuple containing MAE, RMSE, and R2 Score.

    Raises:
        CustomException: If an error occurs during metrics calculation.
    """
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception occurred while evaluating regression metrics')
        raise CustomException(e, sys)

def classification_metrics(true: np.ndarray, predicted: np.ndarray) -> tuple:
    """Calculates classification metrics: accuracy, precision, recall, and F1 score.

    Args:
        true (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, F1 score, and confusion matrix.

    Raises:
        CustomException: If an error occurs during metrics calculation.
    """
    try:
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, average='weighted')  # Use appropriate averaging method
        recall = recall_score(true, predicted, average='weighted')      # Use appropriate averaging method
        f1 = f1_score(true, predicted, average='weighted')              # Use appropriate averaging method
        cm = confusion_matrix(true, predicted)

        return accuracy, precision, recall, f1, cm
    except Exception as e:
        logging.info('Exception occurred while evaluating classification metrics')
        raise CustomException(e, sys)

def print_evaluated_results(xtrain: np.ndarray, ytrain: np.ndarray, xtest: np.ndarray, ytest: np.ndarray, model) -> None:
    """Prints evaluation results for both training and testing datasets.

    Args:
        xtrain (np.ndarray): Training features.
        ytrain (np.ndarray): Training labels.
        xtest (np.ndarray): Testing features.
        ytest (np.ndarray): Testing labels.
        model: Trained model to evaluate.

    Raises:
        CustomException: If an error occurs during printing of evaluated results.
    """
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset for regression metrics
        model_train_mae, model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae, model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        # Evaluate classification metrics
        accuracy, precision, recall, f1, cm = classification_metrics(ytest, ytest_pred)

        # Printing regression results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))

        # Printing classification results
        print('----------------------------------')
        print('Classification Metrics for Test set')
        print("- Accuracy: {:.4f}".format(accuracy))
        print("- Precision: {:.4f}".format(precision))
        print("- Recall: {:.4f}".format(recall))
        print("- F1 Score: {:.4f}".format(f1))
        print("Confusion Matrix:\n", cm)
    
    except Exception as e:
        logging.info('Exception occurred during printing of evaluated results')
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    """Loads an object from a specified file path using dill.

    Args:
        file_path (str): The path from where the object will be loaded.

    Returns:
        object: The loaded object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e, sys)
