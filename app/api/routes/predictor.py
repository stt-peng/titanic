"""Api logic
"""
import json
from typing import Any

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


def get_prediction(data_point: pd.DataFrame) -> Any:
    """Get prediction.

    Args:
        data_point (pd.DataFrame): _description_

    Returns:
        Any: _description_
    """
    return model.predict(data_point, load_wrapper=joblib.load, method="predict")


def get_test_data() -> pd.DataFrame:
    """Load test data.

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_csv("data/test.csv")


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: MachineLearningDataInput):
    """Predict responses.

    Args:
        data_input (MachineLearningDataInput): _description_

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        _type_: _description_
    """

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
    """Healthy method to test connection.

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
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
    """Predict by id list of passengers.

    Args:
        data_input (MachineLearningDataInputList): _description_

    Raises:
        HTTPException: _description_
        ValueError: _description_
        HTTPException: _description_

    Returns:
        _type_: _description_
    """

    if not data_input:
        raise HTTPException(
            status_code=404, detail="'data_input' argument invalid!")
    try:
        id_data = data_input.get_data()
        raw_data = get_test_data()
        if any(id not in raw_data["PassengerId"].to_list() for id in id_data):
            raise ValueError('Check IDs')
        data_point = MachineLearningDataInput(**raw_data[raw_data["PassengerId"].isin(
            id_data)][["Pclass", "Sex", "SibSp", "Parch"]].to_dict(orient="list")).get_dataframe()
        prediction = get_prediction(data_point)

    except Exception as err:
        logger.error(f"Exception: {err}")
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    return MachineLearningResponse(prediction=prediction)
