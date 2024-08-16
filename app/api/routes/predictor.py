import json

import joblib
import pandas as pd
from core.config import INPUT_EXAMPLE
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.prediction import (HealthResponse, MachineLearningDataInput,
                               MachineLearningDataInputList,
                               MachineLearningResponse)
from services.predict import MachineLearningModelHandlerScore as model

router = APIRouter()


# Change this portion for other types of models
# Add the correct type hinting when completed
def get_prediction(data_point: pd.DataFrame):
    return model.predict(data_point, load_wrapper=joblib.load, method="predict")


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: MachineLearningDataInput):

    if not data_input:
        raise HTTPException(
            status_code=404, detail="'data_input' argument invalid!")
    try:
        data_point = data_input.get_dataframe()
        prediction = get_prediction(data_point)

    except Exception as err:
        logger.error(f"Exception: {err}")
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return MachineLearningResponse(prediction=prediction)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
)
async def health():
    is_health = False
    try:
        test_input = MachineLearningDataInput(
            **json.loads(open(INPUT_EXAMPLE, "r").read())
        )
        test_point = test_input.get_dataframe()
        get_prediction(test_point)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")


@router.post(
    "/predict_id_list",
    response_model=MachineLearningResponse,
    name="predict_id_list:get-data",
)
async def predict(data_input: MachineLearningDataInputList):

    if not data_input:
        raise HTTPException(
            status_code=404, detail="'data_input' argument invalid!")
    try:
        id_data = data_input.get_data()
        raw_data = pd.read_csv("app/data/test.csv")
        if any(id not in raw_data["PassengerId"].to_list() for id in id_data):
            raise ValueError('Check IDs')
        data_point = MachineLearningDataInput(**raw_data[raw_data["PassengerId"].isin(
            id_data)][["Pclass", "Sex", "SibSp", "Parch"]].to_dict(orient="list")).get_dataframe()
        prediction = get_prediction(data_point)

    except Exception as err:
        logger.error(f"Exception: {err}")
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return MachineLearningResponse(prediction=prediction)
