import os
import sys
from dataclasses import dataclass

# Importing necessary libraries for machine learning models and evaluation
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, VotingClassifier)  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)  # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from ..exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, model_metrics, print_evaluated_results

@dataclass
class ModelTrainerConfig:
    """Configuration for Model Trainer."""
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """Class for training machine learning models."""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        """Initiates model training process."""
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGB Classifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)
            logging.info(f'Model Report: {model_report}')

            best_model_name, best_model_score = self.get_best_model(model_report)

            # Check if the best model score is satisfactory
            if best_model_score < 0.6:
    
                logging.info('Best model has R2 Score less than 60%')
                
            print(f'Best Model Found: Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found: Model Name: {best_model_name}, R2 Score: {best_model_score}')

            # Hyperparameter tuning for CatBoost
            logging.info('Hyperparameter tuning started for CatBoost')
            cbr = CatBoostClassifier(verbose=False)
            param_dist = {
                'depth': [4, 5, 6, 7, 8, 9, 10],
                'learning_rate': [0.01, 0.02, 0.03, 0.04],
                'iterations': [300, 400, 500, 600]
            }
            rscv = RandomizedSearchCV(cbr, param_dist, scoring='r2', cv=5, n_jobs=-1)
            rscv.fit(x_train, y_train)

            # Print the tuned parameters and score
            print(f'Best CatBoost Parameters: {rscv.best_params_}')
            print(f'Best CatBoost Score: {rscv.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_
            logging.info('Hyperparameter tuning complete for CatBoost')

            # Hyperparameter tuning for KNN
            logging.info('Hyperparameter tuning started for KNN')
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': list(range(2, 31))}
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(x_train, y_train)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters: {grid.best_params_}')
            print(f'Best KNN Score: {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_
            logging.info('Hyperparameter tuning complete for KNN')

            # Create and train Voting Classifier
            logging.info('Voting Classifier training started')
            voting_classifier = VotingClassifier(
                estimators=[('catboost', best_cbr), ('xgb', XGBClassifier()), ('knn', best_knn)],
                weights=[3, 2, 1]
            )
            voting_classifier.fit(x_train, y_train)

            print('Final Model Evaluation:\n')
            print_evaluated_results(x_train, y_train, x_test, y_test, voting_classifier)
            logging.info('Voting Classifier training completed')

            # Save the trained model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=voting_classifier)
            logging.info('Model pickle file saved')

            # Evaluate final model on test data
            y_test_pred = voting_classifier.predict(x_test)
            mae, rmse, r2 = model_metrics(y_test, y_test_pred)
            logging.info(f'Test MAE: {mae}')
            logging.info(f'Test RMSE: {rmse}')
            logging.info(f'Test R2 Score: {r2}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2 

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)

    def get_best_model(self, model_report: dict):
        """Get the best model based on the evaluation report."""
        best_model_name = max(model_report, key=model_report.get)
        best_model_score = model_report[best_model_name]
        return best_model_name, best_model_score
