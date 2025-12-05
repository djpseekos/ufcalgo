# ufcalgo: Probabilistic Modelling and Rating Systems for UFC Fight Outcomes

**ufcalgo** is an end-to-end Python framework for building reproducible datasets, engineered features, and probabilistic models for UFC fight outcomes.  
It provides:

- A complete data pipeline (scraping, cleaning, enriched curated tables)  
- A modular paired-comparison modelling system (Elo / Bradley–Terry–Davidson)  
- Evaluation tools (calibration, scoring rules, diagnostic plots)  
- Backtesting utilities for probability-based decision rules  
- Ongoing development of a **dynamic Bradley–Terry state-space model** using Bayesian inference

The project is structured as a research tool for analysing fighter performance, rating dynamics, model calibration, and decision-making under uncertainty.

---

## **Project Structure**

```
ufcalgo-1/
├── File_Inspector.ipynb             # Notebook for inspecting curated/diagnostic data
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies

├── data/                            # All data artefacts
│   ├── curated/                     # Clean, analysis-ready datasets
│   │   ├── events.csv               # Event metadata
│   │   ├── fighters.csv             # Fighter-level info
│   │   ├── fights.csv               # Fight-level outcomes
│   │   ├── stats_round.csv          # Round-level statistics
│   │   ├── fighters_odds_bfo.csv    # Odds mapped to fighters
│   │   ├── fighter_bfo_links.csv    # Fighter ↔ BFO ID mapping
│   │   ├── elo_history.csv          # Exported fighter rating histories
│   │   └── manifest.json            # Curated dataset manifest
│
│   ├── diagnostics/                 # QA & consistency checks
│   │   ├── bfo_diagnostics.csv      # BestFightOdds scraping diagnostics
│   │   ├── bfo_name_mismatches.csv  # Name mismatch detection
│   │   ├── bfo_backfill_pipeline.csv# Odds backfilling logs
│   │   └── odds_diag_rows.csv       # Row-level odds diagnostics
│
│   ├── lookups/                     # Lookup tables and mappings
│   └── market/                      # Market/odds data (pre-curation)

├── outputs/                         # Model outputs, CV runs, tuning results
│   ├── best_config.json             # Best hyperparameter configuration
│   ├── best_cv_result.json          # Top cross-validation metrics
│   ├── rolling_summary.json         # Rolling-origin evaluation
│   ├── test_metrics.json            # Final test-set performance
│   ├── elo_k_sweep.csv              # Sweep of K-factor values
│   ├── fighter_fight_counts.csv     # Fight count summaries
│   └── tuning_results.csv           # Aggregated tuning results

├── src/
│   ├── cli.py                       # Command-line interface for full pipeline

│   ├── elo_model/                   # Modelling + evaluation framework
│   │   ├── model.py                 # Elo/Bradley–Terry probability model
│   │   ├── feature_builder.py       # Covariate construction
│   │   ├── train.py                 # Parameter fitting (likelihood optimisation)
│   │   ├── tuner.py                 # Hyperparameter sweeps (K, caps, etc.)
│   │   ├── diagnostics.py           # Calibration, scoring rules, summaries
│   │   ├── accuracy_filters.py      # Tools for stratifying accuracy
│   │   ├── backtest_strategies.py   # Strategy definitions & backtesting
│   │   ├── export_elo_history.py    # Export fighter rating trajectories
│   │   ├── experience_curve.py      # Experience/learning-curve analysis
│   │   ├── debut_check.py           # Handling debut / low-history fighters
│   │   ├── fighter_counts.py        # Per-fighter fight statistics
│   │   ├── time_utils.py            # Time calculations (rest days, etc.)
│   │   └── helpers.py               # Shared utilities for modelling pipeline

│   └── scraper/                     # Web scraping + odds integration
│       ├── ufcstats/                # UFCStats scrapers
│       │   ├── ufcstats_events.py       # Scrape events
│       │   ├── ufcstats_fighters.py     # Scrape fighter profiles
│       │   └── ufcstats_fight_rounds.py # Scrape round-level stats
│
│       ├── bestfightodds/           # BestFightOdds scraping/wrangling
│       │   ├── scraping/                # HTML scraping & mapping
│       │   │   ├── scrape_odds_bfo.py
│       │   │   ├── backfill_odds.py
│       │   │   └── map_fighters_bfo.py
│       │   ├── wrangling/               # Clean & enrich odds
│       │   │   ├── augment_fights_with_bfo_odds.py
│       │   │   ├── convert_dates_iso.py
│       │   │   └── filter_odds_by_fights.py
│       │   └── diagnostics/             # Odds QA
│       │       ├── backfill_diagnostics.py
│       │       └── name_mismatches.py
│
│       └── common/                   # Shared scraping utilities
│           ├── http.py               # HTTP client with retries/throttling
│           ├── parse.py              # HTML parsing helpers
│           ├── io.py                 # JSON/CSV I/O utilities
│           ├── names.py              # Fighter-name normalisation & matching
│           └── ids.py                # ID generation & mapping helpers
```

---

## **Capabilities**

### **1. Data Collection & Integration**
- Scrapes:
  - Fighter profiles
  - Event metadata
  - Fight outcomes
  - Round-level striking & grappling statistics  
- Gathers opening/closing odds from *BestFightOdds*.  
- Performs consistency checks, missing-data detection, and automated enrichment (timezones, altitude, weight misses, etc.).  
- Produces analysis-ready curated tables under `data/curated/`.

---

### **2. Feature Engineering**
- Constructs fight-level covariates:
  - Rest time / activity level  
  - Experience indicators  
  - Weight-class transitions  
  - Aggregated round-level statistics  
- Supports extensible custom features for modelling or backtesting.

---

### **3. Probabilistic Modelling**
Implements a flexible paired-comparison modelling framework:

- **Elo–Bradley–Terry hybrid model** for win probabilities  
- 3-outcome BTD-style probability formulation  
- Covariate-dependent K-factors  
- Likelihood-based estimation using numerical optimisation  
- Outputs calibrated win-probability predictions  
- Includes tools for:
  - Brier score  
  - Log-loss  
  - Reliability diagrams  
  - Class-stratified diagnostics  

---

### **4. Backtesting & Evaluation**
- Rolling-origin evaluation  
- Accuracy breakdowns by:
  - weight class  
  - recency  
  - confidence interval bin  
- ROI / betting-strategy backtesting (fixed-stake, Kelly variants, thresholding rules)  
- JSON and CSV summaries for reproducibility

---

### **5. Ongoing Research: Dynamic Bradley–Terry (Bayesian State-Space Model)**
Actively in development (separate module):

- Time-varying latent fighter strengths  
- Gaussian-evolution state dynamics  
- Penalised-spline or learned covariate effects  
- Hierarchical shrinkage priors  
- NUTS-based Bayesian inference (PyMC)  
- Posterior predictive checks and simulation studies  

(*Not yet integrated into CLI workflow. Experimental stage.*)

---

## **Usage**

### **Run the entire pipeline**

```bash
python -m src.cli events
```

### **Run individual components**

```bash
python -m src.cli events
python -m src.cli fights
python -m src.cli rounds
python -m src.cli fighters
python -m src.cli context
python -m src.cli odds
```

---

## **Installation**

```bash
git clone https://github.com/djpseekos/ufcalgo
cd ufcalgo
pip install -r requirements.txt
```

---

## **Notes**
- Web scrapers depend on UFCStats layout and may require maintenance if the site structure changes.  
- Dynamic Bradley–Terry model is under active development and remains separate from the production pipeline.

---

## **Author**
**Perseus Georgiadis, University of Bath**  
Contact: perseus.georgiadis@gmail.com
