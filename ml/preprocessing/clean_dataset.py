# -*- coding: utf-8 -*-
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger


class Pipeline:
    def __init__(self, file_name: str, input_filepath: str, output_filepath: str) -> None:
        self.file_name = file_name
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def read_data(self) -> None:
        logger.info(f"Start reading data.")
        self.data = pd.read_csv(os.path.join(
            self.input_filepath, self.file_name))

    def select_train_columns(self, features: list = ["Pclass", "Sex", "SibSp", "Parch"]) -> None:
        logger.info(f"Start selecting train columns.")
        self.x = self.data[features]

    def select_target_column(self, target_col_name: str = "Survived") -> None:
        logger.info(f"Start selecting target column.")
        try:
            self.y = self.data[[target_col_name]]
        except KeyError:
            self.y = pd.DataFrame([])

    def concat_data(self) -> None:
        logger.info(f"Concat data.")
        self.prep_data = pd.concat([self.x, self.y], axis=1)
        logger.info(self.prep_data)

    def write(self) -> None:
        logger.info(f"Start writing data.")
        self.prep_data.to_csv(os.path.join(
            self.output_filepath, self.file_name), index=False)

    def run(self) -> pd.DataFrame:
        logger.info("Start making interim dataset.")
        self.read_data()
        self.select_train_columns()
        self.select_target_column()
        self.concat_data()
        self.write()


@click.command()
@click.argument("file_name", default="data.csv", type=click.Path())
@click.argument("input_filepath", default="data/raw", type=click.Path(exists=True))
@click.argument("output_filepath", default="data/interim", type=click.Path(exists=True))
def main(file_name, input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info(f"Read from {input_filepath}, write to {output_filepath}.")

    pipeline = Pipeline(
        file_name=file_name, input_filepath=input_filepath, output_filepath=output_filepath)
    pipeline.run()


if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
