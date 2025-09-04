# src/elo_model/debut_check.py
from collections import defaultdict
from .train import EloBTDTrainer

def debut_vs_status(rows):
    counts = {"double_debut": 0, "prospect_vs_vet": 0, "vet_vs_vet": 0}
    fight_counts = defaultdict(int)

    for f in rows:
        ci, cj = fight_counts[f.i], fight_counts[f.j]
        if ci == 0 and cj == 0:
            counts["double_debut"] += 1
        elif ci == 0 or cj == 0:
            counts["prospect_vs_vet"] += 1
        else:
            counts["vet_vs_vet"] += 1
        # increment AFTER the fight
        fight_counts[f.i] += 1
        fight_counts[f.j] += 1

    return counts

if __name__ == "__main__":
    trainer = EloBTDTrainer()
    rows = trainer.load_data("data/curated/fights.csv", "data/curated/events.csv")
    rows = trainer.build_features(rows)

    counts = debut_vs_status(rows)
    print("Fight type counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")