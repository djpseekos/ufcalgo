# src/elo_model/accuracy_filters.py
from collections import defaultdict
import pandas as pd
import numpy as np
from .model import btd_probs, K_value

def accuracy_with_experience(fights, params, K_cap=128, min_fights=5):
    """
    Replay Elo-BTD with params, but only evaluate accuracy/logloss
    on fights where BOTH fighters already had ≥min_fights past bouts.
    """
    a, Kmin, Kmax, nu = params[:4]
    beta = params[4:]

    # Ratings & counters
    R_class = {}
    R_global = {}
    class_means, class_counts = {}, {}
    fight_counts = defaultdict(int)  # fighter_id -> number of past fights

    preds, outcomes = [], []

    for f in fights:
        ki, kj = (f.i, f.wc), (f.j, f.wc)
        # init ratings if not present
        for k, fid in [(ki, f.i), (kj, f.j)]:
            if k not in R_class:
                mu_c = class_means.get(f.wc, 1500.0)
                R_class[k] = a*R_global.get(fid,1500.0)+(1-a)*mu_c
                class_means[f.wc] = (class_means.get(f.wc,0.0)*class_counts.get(f.wc,0)+R_class[k])/(class_counts.get(f.wc,0)+1)
                class_counts[f.wc] = class_counts.get(f.wc,0)+1

        Ri, Rj = R_class[ki], R_class[kj]
        p_i, p_d, p_j = btd_probs(Ri, Rj, nu)

        # Only count this fight if BOTH had at least `min_fights` already
        if fight_counts[f.i] >= min_fights and fight_counts[f.j] >= min_fights:
            pred = np.argmax([p_i, p_d, p_j])
            true = {"i":0,"d":1,"j":2}[f.outcome]
            preds.append(pred)
            outcomes.append(true)

        # Update ratings regardless (so counts are realistic)
        if f.outcome == "i": S_i = 1.0
        elif f.outcome == "j": S_i = 0.0
        else: S_i = 0.5
        S_hat_i = p_i + 0.5*p_d
        Kt = min(K_value(Kmin, Kmax, beta, f.x), K_cap)
        R_class[ki] = Ri + Kt*(S_i - S_hat_i)
        R_class[kj] = Rj + Kt*((1-S_i) - (p_j+0.5*p_d))

        # Increment fight counters after fight
        fight_counts[f.i] += 1
        fight_counts[f.j] += 1

    if not preds: 
        return {"accuracy": None, "n_eval": 0}
    acc = float(np.mean(np.array(preds)==np.array(outcomes)))
    return {"accuracy": acc, "n_eval": len(preds)}