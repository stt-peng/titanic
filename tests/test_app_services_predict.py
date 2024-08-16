
from unittest.mock import MagicMock, patch

import pytest
from core.config import MODEL_NAME, MODEL_PATH
from core.errors import ModelLoadException, PredictException
from services.predict import MachineLearningModelHandlerScore


@pytest.fixture
def mock_load_wrapper():
    return MagicMock()


@pytest.fixture
def mock_model():
    return MagicMock()


class TestMachineLearningModelHandlerScore:

    @staticmethod
    def test_predict_calls_method_if_exists(mock_load_wrapper, mock_model):
        # Setup
        MachineLearningModelHandlerScore.model = mock_model
        mock_model.predict.return_value = "prediction"

        input_data = "input"
        method = "predict"

        # Test
        result = MachineLearningModelHandlerScore.predict(
            input_data, load_wrapper=mock_load_wrapper, method=method)

        # Assert
        mock_model.predict.assert_called_once_with(input_data)
        assert result == "prediction"

    @staticmethod
    @patch.object(MachineLearningModelHandlerScore, "get_model")
    def test_predict_raises_exception_if_method_does_not_exist(mock_method, mock_load_wrapper):
        # Setup
        mock_method.return_value = []
        method = "non_existent_method"

        # Test and Assert
        with pytest.raises(PredictException, match=f"'{method}' attribute is missing"):
            MachineLearningModelHandlerScore.predict(
                "input", load_wrapper=mock_load_wrapper, method=method)

    @staticmethod
    @patch.object(MachineLearningModelHandlerScore, "load")
    def test_get_model_loads_model_if_none_exists(mock_load):
        # Setup
        MachineLearningModelHandlerScore.model = None
        mock_load.return_value = "loaded_model"

        load_wrapper = 'test'

        # Test
        machine_learning_model_handler_score = MachineLearningModelHandlerScore()
        model = machine_learning_model_handler_score.get_model(
            load_wrapper=load_wrapper)

        # Assert
        assert model == "loaded_model"
        mock_load.assert_called_once_with(load_wrapper)
        assert MachineLearningModelHandlerScore.model == "loaded_model"

    @staticmethod
    @patch("services.predict.os.path.exists")
    @patch("services.predict.logger")
    def test_load_raises_exception_if_path_does_not_exist(mock_logger, mock_exists, mock_load_wrapper):
        # Setup
        mock_exists.return_value = False

        # Test and Assert
        with pytest.raises(FileNotFoundError, match=f"Machine learning model at {MODEL_PATH}{MODEL_NAME} not exists!"):
            MachineLearningModelHandlerScore.load(
                load_wrapper=mock_load_wrapper)
        mock_logger.error.assert_called_once()

    @staticmethod
    @patch("services.predict.os.path.exists")
    def test_load_raises_exception_if_model_cannot_be_loaded(mock_exists, mock_load_wrapper):
        # Setup
        mock_exists.return_value = True
        mock_load_wrapper.return_value = None

        # Test and Assert
        with pytest.raises(ModelLoadException, match=f"Model None could not load!"):
            MachineLearningModelHandlerScore.load(
                load_wrapper=mock_load_wrapper)
