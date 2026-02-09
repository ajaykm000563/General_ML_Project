import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models




@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
      try:
        logging.info("Splitting training and test input data")

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models = {
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "Logistic Regression": LogisticRegression(),
            "Linear Regression": LinearRegression()
        }

        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models
        )

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found with score greater than 0.6", sys)

        logging.info(
            f"Best found model is {best_model_name} with r2 score: {best_model_score}"
        )

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)

        return r2_square

      except Exception as e:
        raise CustomException(e, sys)

