from pydantic import BaseModel
from typing import Optional

class ScreeningInput(BaseModel):
    # Behavior scores
    A1:  int
    A2:  int
    A3:  int
    A4:  int
    A5:  int
    A6:  int
    A7:  int
    A8:  int
    A9:  int
    A10: int

    # Demographics
    age:       int
    gender:    int
    jundice:   int
    austim:    int
    ethnicity: int
    result:    int

    # Doctor scores (from synthetic_asd_dataset.csv)
    D1:  int
    D2:  int
    D3:  int
    D4:  int
    D5:  int
    D6:  int
    D7:  int
    D8:  int
    D9:  int
    D10: int


class PredictionOutput(BaseModel):
    prediction:          str
    asd_probability:     float
    non_asd_probability: float
    risk_level:          str
    risk_level_num:      int = 1
    top_features:        dict[str, float]
    shap_plot:           Optional[str] = None
    lime_plot:           Optional[str] = None