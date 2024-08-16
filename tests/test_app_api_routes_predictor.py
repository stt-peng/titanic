from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

@patch("services.predict.MODEL_NAME", "dummy_model.pkl")
class TestApp():

    @staticmethod
    def test_health(mock_model):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {'status': True}

    @staticmethod
    def test_predict_good_input(mock_model):
        response = client.post(
            "/api/v1/predict", json={"Pclass": [3], "SibSp": [0], "Parch": [0], "Sex": ["male"]})
        assert response.status_code == 200
        assert response.json() == {"prediction": [0]}

    @staticmethod
    def test_predict_bad_input(mock_model):
        response = client.post(
            "/api/v1/predict", json={"Pclass": [3]})
        assert response.status_code == 422

    @staticmethod
    @patch("api.routes.predictor.get_test_data")
    def test_predict_id_list_good_input(mock_model, mock_get_test_data):
        mock_get_test_data.return_value = return_value = pd.read_csv(
            "app/data/test.csv")

        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": [900, 901]})
        assert response.status_code == 200
        assert response.json() == {"prediction": [1, 0]}

    @staticmethod
    @patch("api.routes.predictor.get_test_data")
    def test_predict_id_list_bad_input(mock_model, mock_get_test_data):
        mock_get_test_data.return_value = return_value = pd.read_csv(
            "app/data/test.csv")

        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": ["asd"]})
        assert response.status_code == 422

    @staticmethod
    @patch("api.routes.predictor.get_test_data")
    def test_predict_id_list_bad_id(mock_get_test_data, mock_model):
        mock_get_test_data.return_value = return_value = pd.read_csv(
            "app/data/test.csv")

        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": [1, 2]})
        assert response.status_code == 500
        assert response.json() == {"detail": "Exception: Check IDs"}
