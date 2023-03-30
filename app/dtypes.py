from pydantic import BaseModel
from typing import Dict, List

class Ticker(BaseModel):
    ticker: str

class ExperimentParametersXGBoost(BaseModel):
    ticker: str
    target: str
    model_type: str
    model_class: str
    name: str
    current_date: str
    features: List[str]
    horizon : int
    window : int
    max_evals : int
    scores : List[str]
    parallelism : int
    max_depth: Dict[str, int]
    n_estimators: Dict[str, int]
    reg_alpha: Dict[str, float]
    max_delta_step: Dict[str, float]
    min_split_loss: Dict[str, float]
    learning_rate: Dict[str, float]
    min_child_weight: Dict[str, float]
    scale_pos_weight: Dict[str, float]

class ExperimentParametersAlgoMA(BaseModel):
    ticker: str
    target: str
    model_type: str
    model_class: str
    name: str
    current_date: str
    features: List[str]
    horizon : int
    window : int
    max_evals : int
    scores : List[str]
    parallelism : int
    short_term_ma : Dict[str, int]
    long_term_ma: Dict[str, int]

class Model(BaseModel):
    model_name: str

class ModelRequest(BaseModel):
    model_name: str
    data: Dict[str, Dict[str, float]]