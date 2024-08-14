import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class Model():
    def __init__(self, train_path: str = "data/raw/train.csv", test_path: str = "data/raw/test.csv") -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.features = ["Pclass", "Sex", "SibSp", "Parch"]

    def load_data(self) -> None:
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def train(self) -> None:
        y = self.train_data["Survived"]

        X = pd.get_dummies(self.train_data[self.features])

        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)

        self.model = model

    def get_test_predictions(self) -> None:
        X_test = pd.get_dummies(self.test_data[self.features])
        predictions = self.model.predict(X_test)

        return pd.DataFrame({'PassengerId': self.test_data.PassengerId, 'Survived': predictions})

    def save_model(self, model_name: str = os.getenv('MODEL_NAME'), model_path: str = os.getenv('MODEL_PATH')) -> None:
        joblib.dump(self.model, os.path.join(model_path, model_name))


if __name__ == "__main__":
    titanic_model = Model()
    titanic_model.load_data()
    titanic_model.train()
    predictions = titanic_model.get_test_predictions()
    titanic_model.save_model()

    print(predictions)
