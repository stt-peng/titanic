# -*- coding: utf-8 -*-
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from loguru import logger


class Pipeline:
    def __init__(self, file_name: str, input_filepath: str, output_filepath: str, is_train: bool = True) -> None:
        self.file_name = file_name
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.is_train = is_train

    def read_data(self) -> None:
        logger.info(f"Start reading data.")
        self.data = pd.read_csv(os.path.join(
            self.input_filepath, self.file_name))

    def get_features(self) -> None:
        logger.info(f"Get features.")
        if self.is_train:
            self.x = self.data[self.data.columns[:-1]]
            self.y = self.data[self.data.columns[-1]]
        else:
            self.x = self.data
            self.y = pd.DataFrame([])

    def get_dummies(self) -> None:
        logger.info(f"Get features.")
        self.x = pd.get_dummies(self.x)

    def concat_data(self) -> None:
        logger.info(f"Concat data.")
        self.feat_data = pd.concat([self.x, self.y], axis=1)

    def write(self) -> None:
        logger.info(f"Start writing data.")
        self.feat_data.to_csv(os.path.join(
            self.output_filepath, self.file_name), index=False)

    def run(self) -> None:
        logger.info("Start building features.")
        self.read_data()
        self.get_features()
        self.get_dummies()
        self.concat_data()
        self.write()


@click.command()
@click.argument("file_name", default="data.csv", type=click.Path())
@click.argument("input_filepath", default="data/interim", type=click.Path(exists=True))
@click.argument("output_filepath", default="data/processed", type=click.Path(exists=True))
def main(file_name, input_filepath, output_filepath):
    """Runs data processing scripts to turn cleaned data from (../interim) into
    training data ready to be trained (saved in ../processed).
    """
    logger.info(f"Read from {input_filepath}, write to {output_filepath}.")
    pipeline = Pipeline(
        file_name=file_name, input_filepath=input_filepath, output_filepath=output_filepath)
    pipeline.run()


if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
