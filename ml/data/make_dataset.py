# -*- coding: utf-8 -*-
import zipfile
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger


class Pipeline:
    def __init__(self, output_filepath: str) -> None:
        self.output_filepath = output_filepath

    def download_data(self) -> None:
        logger.info(f"Start downloading data.")
        api = KaggleApi()
        api.authenticate()

        api.competition_download_files(
            competition="titanic", path=self.output_filepath)

    def extract_data(self) -> None:
        logger.info(f"Start unzipping data.")
        with zipfile.ZipFile(f"{self.output_filepath}/titanic.zip", 'r') as zip_ref:
            zip_ref.extractall(self.output_filepath)

    def run(self):
        logger.info("Start making raw dataset.")
        self.download_data()
        self.extract_data()


@click.command()
@click.argument("output_filepath", default="data/raw", type=click.Path(exists=True))
def main(output_filepath):
    """Runs data scripts to create raw data.
    """
    logger.info(f"Read from kaggle, write to {output_filepath}.")
    pipeline = Pipeline(output_filepath=output_filepath)
    pipeline.run()


if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # pylint: disable = no-value-for-paramete
    main()
