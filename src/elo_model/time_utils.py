# src/elo_model/time_utils.py
import math
from .helpers import is_decision

def actual_elapsed_seconds(method: str, scheduled_rounds: float, end_round: float, end_time_sec: float) -> int:
    """Decision -> full scheduled time; else (end_round-1)*5*60 + end_time_sec."""
    sr = 3.0 if (scheduled_rounds is None or (isinstance(scheduled_rounds, float) and math.isnan(scheduled_rounds))) else float(scheduled_rounds)
    if is_decision(method):
        return int(sr * 5 * 60)
    r = 1 if (end_round is None or (isinstance(end_round, float) and math.isnan(end_round))) else int(end_round)
    t = 0 if (end_time_sec is None or (isinstance(end_time_sec, float) and math.isnan(end_time_sec))) else int(end_time_sec)
    return max(0, (r - 1) * 5 * 60 + t)