from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel


class MachineLearningResponse(BaseModel):
    prediction: List[int]


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    Pclass: List[int]
    SibSp: List[int]
    Parch: List[int]
    Sex: List[str]

    def get_dataframe(self):
        return pd.DataFrame(
            {
                "Pclass": self.Pclass,
                "SibSp": self.SibSp,
                "Parch": self.Parch,
                "Sex_female": np.array(self.Sex) == 'female',
                "Sex_male": np.array(self.Sex) == 'male'
            }
        )


class MachineLearningDataInputList(BaseModel):
    PassengerId: List[int]

    def get_data(self):
        return self.PassengerId
