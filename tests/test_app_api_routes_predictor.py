from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestApp():

    @staticmethod
    def test_health():
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {'status': True}

    @staticmethod
    def test_predict_good_input():
        response = client.post(
            "/api/v1/predict", json={"Pclass": [3], "SibSp": [0], "Parch": [0], "Sex": ["male"]})
        assert response.status_code == 200
        assert response.json() == {"prediction": [0]}

    @staticmethod
    def test_predict_bad_input():
        response = client.post(
            "/api/v1/predict", json={"Pclass": [3]})
        assert response.status_code == 422

    @staticmethod
    def test_predict_id_list_good_input():
        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": [900, 901]})
        assert response.status_code == 200
        assert response.json() == {"prediction": [1, 0]}

    @staticmethod
    def test_predict_id_list_bad_input():
        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": ["asd"]})
        assert response.status_code == 422

    @staticmethod
    def test_predict_id_list_bad_id():
        response = client.post(
            "/api/v1/predict_id_list", json={"PassengerId": [1, 2]})
        assert response.status_code == 500
        assert response.json() == {"detail": "Exception: Check IDs"}
