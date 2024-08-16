from dotenv import find_dotenv, load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from ml.features import build_features
from ml.model import train_model
from ml.preprocessing import clean_dataset

load_dotenv(find_dotenv())

from ml.data import make_dataset

class Pipeline:
    def __init__(self) -> None:
        pass

    def run(self) -> None:
        make_dataset_pipe = make_dataset.Pipeline(output_filepath="data/raw")
        make_dataset_pipe.run()

        for file in ["train.csv", "test.csv"]:
            clean_dataset_pipe = clean_dataset.Pipeline(
                file_name=file, input_filepath="data/raw", output_filepath="data/interim")
            clean_dataset_pipe.run()

            build_features_pipe = build_features.Pipeline(
                file_name=file, input_filepath="data/interim", output_filepath="data/processed", is_train=file == "train.csv")
            build_features_pipe.run()

        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=1)

        train_model_pipe = train_model.Pipeline(train_data_name="train.csv", test_data_name="test.csv", input_data_path="data/processed",
                                                output_data_path="data/results", model_name="model.pkl", model_path="ml/model/", model=model)
        train_model_pipe.run()

        logger.info("Pipeline finished.")


if __name__ == "__main__":

    pipeline = Pipeline()
    pipeline.run()
