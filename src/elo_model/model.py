# src/elo_model/model.py
import math
import numpy as np
from dataclasses import dataclass

LOG10 = math.log(10.0)

@dataclass
class FightRow:
    i: str
    j: str
    outcome: str   # 'i', 'j', or 'd'
    wc: str
    date: object   # pandas.Timestamp
    scheduled_rounds: int
    method: str
    end_round: int
    end_time_sec: int
    is_title: int
    judge_scores: str
    x: np.ndarray | None = None  # features to be filled later

def logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))

def btd_probs(Ri: float, Rj: float, nu: float) -> tuple[float, float, float]:
    """Bradley–Terry–Davidson probabilities."""
    li = (Ri / 400.0) * LOG10
    lj = (Rj / 400.0) * LOG10
    ai, aj = math.exp(li), math.exp(lj)
    denom = ai + aj + nu * math.sqrt(ai * aj)
    p_i = ai / denom
    p_j = aj / denom
    p_d = 1.0 - p_i - p_j
    return p_i, p_d, p_j

def K_value(Kmin: float, Kmax: float, beta: np.ndarray, x: np.ndarray) -> float:
    eta = float(np.dot(beta, x))
    return Kmin + (Kmax - Kmin) * logistic(eta)