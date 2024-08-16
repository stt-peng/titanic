import os
from typing import Any

import click
import joblib
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class Pipeline:
    def __init__(self, train_data_name: str, test_data_name: str, input_data_path: str, output_data_path: str, model_name: str, model_path: str, model: Any) -> None:
        self.train_data_name = train_data_name
        self.test_data_name = test_data_name
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.model_name = model_name
        self.model_path = model_path
        self.model = model

    def read_data(self) -> None:
        logger.info(f"Start reading data.")
        self.train_data = pd.read_csv(os.path.join(
            self.input_data_path, self.train_data_name))
        self.test_data = pd.read_csv(os.path.join(
            self.input_data_path, self.test_data_name))

    def get_features(self) -> None:
        logger.info(f"Get features.")
        self.x_train = self.train_data[self.train_data.columns[:-1]]
        self.y_train = self.train_data[self.train_data.columns[-1]]
        self.x_test = self.test_data

    def split_data(self) -> None:
        self.x_split_train, self.x_split_test, self.y_split_train, self.y_split_test = train_test_split(
            self.x_train, self.y_train, test_size=0.1, random_state=2024)

    def train(self) -> None:
        logger.info(f"Start training model.")
        self.model.fit(self.x_split_train, self.y_split_train)

    def get_test_prediction(self, data_to_predict: pd.DataFrame) -> None:
        logger.info(f"Predict data.")
        return self.model.predict(data_to_predict)

    def write_test_prediction(self, data_to_write: pd.DataFrame, file_name: str) -> None:
        logger.info(f"Start writing data.")
        pd.DataFrame(data_to_write).to_csv(os.path.join(
            self.output_data_path, file_name), index=False)

    def save_model(self) -> None:
        logger.info(f"Start saving model.")
        joblib.dump(self.model, os.path.join(self.model_path, self.model_name))

    def run(self) -> None:
        logger.info(f"Start train pipeline.")
        self.read_data()
        self.get_features()
        self.split_data()
        self.train()
        x_test_pred = self.get_test_prediction(data_to_predict=self.x_test)
        self.write_test_prediction(
            data_to_write=x_test_pred, file_name=self.test_data_name)
        y_split_test_pred = self.get_test_prediction(
            data_to_predict=self.x_split_test)
        self.write_test_prediction(
            data_to_write=y_split_test_pred, file_name="split_test_pred.csv")
        self.write_test_prediction(
            data_to_write=self.y_split_test, file_name="split_test_orig.csv")
        self.save_model()

        print(classification_report(self.y_split_test, y_split_test_pred))


@ click.command()
@ click.argument("train_data_name", default="train_data.csv", type=click.Path())
@ click.argument("test_data_name", default="test_data.csv", type=click.Path())
@ click.argument("input_data_path", default="data/processed", type=click.Path(exists=True))
@ click.argument("output_data_path", default="data/results", type=click.Path(exists=True))
@ click.argument("model_name", default="model.pkl", type=click.Path())
@ click.argument("model_path", default="ml/model", type=click.Path(exists=True))
def main(train_data_name, test_data_name, input_data_path, output_data_path, model_name, model_path):
    """Runs train model.
    """
    logger.info(f"Read from {input_data_path}, write to {model_path}.")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=1)
    pipeline = Pipeline(train_data_name=train_data_name, test_data_name=test_data_name, input_data_path=input_data_path,
                        output_data_path=output_data_path, model_name=model_name, model_path=model_path, model=model)
    pipeline.run()


if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
