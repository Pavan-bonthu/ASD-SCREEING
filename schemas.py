from pydantic import BaseModel
from typing import Optional

class ScreeningInput(BaseModel):
    A1_Score:  int
    A2_Score:  int
    A3_Score:  int
    A4_Score:  int
    A5_Score:  int
    A6_Score:  int
    A7_Score:  int
    A8_Score:  int
    A9_Score:  int
    A10_Score: int
    age:       int
    gender:    int
    jundice:   int
    austim:    int
    ethnicity: int
    result:    int

class PredictionOutput(BaseModel):
    prediction:          str
    asd_probability:     float
    non_asd_probability: float
    risk_level:          str
    top_features:        dict
    shap_plot:           Optional[str] = None
    lime_plot:           Optional[str] = None