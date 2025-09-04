# src/elo_model/helpers.py
import re

def is_decision(method: str) -> bool:
    if not isinstance(method, str): return False
    m = method.strip().lower()
    return "decision" in m  # catches 'technical decision', 'decision - unanimous', etc.

def is_ko_tko(method: str) -> bool:
    if not isinstance(method, str): return False
    m = method.strip().lower()
    return ("ko" in m) or ("tko" in m)

def parse_judge_margin(judge_scores: str):
    """Return (sum_abs_margins, J) by parsing 'NN - NN' patterns; (0,0) if none."""
    if not isinstance(judge_scores, str): return (0, 0)
    totals = re.findall(r'(\d+)\s*[-–]\s*(\d+)', judge_scores)
    if not totals: return (0, 0)
    margins = [abs(int(a) - int(b)) for a, b in totals]
    return sum(margins), len(margins)