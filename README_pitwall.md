# pitwall-analytics

Formula 1 end-to-end data science and analytics project covering the 2025 full season and the 2026 season in progress. The project builds ML models for race outcome prediction, clusters drivers into archetypes, detects anomalous race performances, and implements a GenAI retrieval system for natural language Q&A on F1 regulations and standings.

---

## Repository Contents

```
pitwall-analytics/
|
|-- F1_2026_DataScience_Project_v2.ipynb   Main analysis notebook (14 sections)
|
|-- f1_2025_race_results.csv               All 24 races, 22 drivers, full 2025 season
|-- f1_2026_results_so_far.csv             Australia GP + China Sprint/Qualifying
|-- f1_2026_driver_profiles.csv            22 drivers, career stats + 2026 metrics
|-- f1_2026_team_specs.csv                 11 teams, technical specs and engine data
|-- f1_2026_tracks.csv                     24 circuits, power sensitivity and overtake data
|-- f1_2026_championship_prediction.csv    ML-predicted final standings
```

---

## What the Notebook Covers

| Section | Topic |
|---------|-------|
| 1 | Imports and data loading |
| 2 | 2025 season exploratory data analysis |
| 3 | 2026 season current state analysis |
| 4 | Statistical hypothesis tests |
| 5 | Feature engineering |
| 6 | ML - Race winner predictor (Random Forest) |
| 7 | ML - Championship predictor (Gradient Boosting Regression) |
| 8 | ML - Podium and DNF classifiers |
| 9 | K-Means clustering - driver archetypes |
| 10 | Isolation Forest - anomaly detection |
| 11 | GenAI RAG - F1 policy Q&A system (TF-IDF) |
| 12 | Business intelligence dashboard (6-panel) |
| 13 | Key insights and predictions |
| 14 | Save all outputs |

---

## Datasets

### f1_2025_race_results.csv
528 rows, 25 columns. One row per driver per race across all 24 rounds of the 2025 season.

Key columns: `driver`, `team`, `grand_prix`, `circuit`, `grid_position`, `finish_position`, `race_points`, `sprint_points`, `dnf`, `pit_stops`, `tyre_compound_primary`, `weather`, `air_temp_c`, `track_temp_c`, `lap_time_gap_sec`, `fastest_lap_bonus`, `has_sprint`

### f1_2026_results_so_far.csv
66 rows, 19 columns. Covers Australia GP (race), China Sprint, and China Qualifying.

Key columns: `session_type`, `driver`, `team`, `engine_supplier`, `position`, `points_earned`, `cumulative_points`, `active_aero`, `overtake_mode_used`, `recharge_mode_laps`

### f1_2026_driver_profiles.csv
22 rows, 20 columns. One row per driver with career statistics and 2026-specific performance ratings.

Key columns: `driver`, `team_2026`, `engine_supplier_2026`, `age_2026`, `career_wins`, `career_poles`, `world_championships`, `car_performance_rating_2026`, `driver_skill_rating`, `active_aero_adaptation_score`, `energy_management_rating`, `overtake_mode_efficiency`, `2025_championship_pts`, `2026_pts_so_far`, `is_rookie_2026`, `team_changed_2026`

### f1_2026_team_specs.csv
11 rows, 25 columns. Technical specifications for all constructors entering the 2026 season.

Key columns: `team`, `engine`, `chassis`, `car_weight_kg`, `engine_power_kw`, `ers_deployment_kw`, `car_perf_rating`, `preseason_lap_rank`, `tyre_deg_rating`, `overtake_mode_efficiency`, `straight_line_speed_kmh`, `new_engine_2026`, `notable_changes`

### f1_2026_tracks.csv
24 rows, 16 columns. Circuit characteristics for every round on the 2026 calendar.

Key columns: `circuit`, `gp`, `type`, `length_km`, `laps`, `active_aero_zones`, `overtake_difficulty`, `power_sensitivity`, `tyre_wear`, `downforce_req`, `historical_winner_advantage`

---

## Machine Learning Models

### Race Winner Predictor (Random Forest)
Predicts whether a driver will win a given race.

- Target: `won` (binary, 1 if finish position = 1)
- Train/test split: 80/20, stratified
- Class balancing: `class_weight='balanced'`
- AUC-ROC: ~0.76

Features used:
```
grid_position, team_enc, nat_enc, weather_enc, tyre_enc,
pit_stops, air_temp_c, track_temp_c, power_sensitivity,
overtake_difficulty, tyre_wear_enc, down_enc
```

Top predictive features (by importance): `track_temp_c`, `nat_enc`, `air_temp_c`, `team_enc`, `grid_position`

### 2026 Championship Predictor (Gradient Boosting Regression)
Projects the final 2026 points total for all 22 drivers.

- Target: composite projected points (0-420)
- Features: car performance rating, driver skill rating, active aero adaptation, energy management, overtake mode efficiency, 2025 championship points, career wins, world championships
- Preprocessing: StandardScaler

Predicted 2026 top 3:
```
1. George Russell    (Mercedes)
2. Charles Leclerc   (Ferrari)
3. Lewis Hamilton    (Ferrari)
```

### Podium Predictor (Gradient Boosting Classifier)
Predicts probability of a top-3 finish.

- Target: `podium` (binary, finish position <= 3)
- Same 12 features as race winner model

### DNF Risk Predictor (Gradient Boosting Classifier)
Flags retirement risk before a race.

- Target: `dnf` (binary)
- Fix applied: `GradientBoostingClassifier` does not accept `class_weight`. Imbalance is handled by passing `sample_weight` to `.fit()` via `compute_sample_weight`:

```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(class_weight='balanced', y=ytr_d)
gb_dnf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_dnf.fit(Xtr_d, ytr_d, sample_weight=sample_weights)
```

---

## Statistical Tests

| Test | Variables | Result |
|------|-----------|--------|
| Independent t-test | Grid position: top teams vs bottom teams | Significant (p < 0.05) |
| Pearson correlation | 2025 championship pts vs 2026 car rating | Reported in notebook |
| One-way ANOVA | Race points by weather condition | Reported in notebook |
| Chi-square | Team vs podium finish | chi2 = 34.21, p < 0.001 — significant |
| Spearman correlation | Grid position vs finish position (non-DNF races only) | Strong monotonic relationship |

Note on the Spearman fix: the mask `~df25['dnf']` must be applied to both arrays before passing to `stats.spearmanr`. Using `df25.loc[no_dnf, 'column']` ensures both arrays are the same length (516 rows, excluding DNFs).

---

## Clustering - Driver Archetypes

K-Means with K=4 applied to 7 MinMax-scaled features:
`age_2026`, `career_wins`, `world_championships`, `car_performance_rating_2026`, `driver_skill_rating`, `active_aero_adaptation_score`, `energy_management_rating`

PCA (2 components) used for visualization.

| Archetype | Criteria | Example Drivers |
|-----------|----------|-----------------|
| Elite Champion | career wins > 20 | Hamilton, Verstappen, Alonso |
| Race Winner | career wins > 5 | Russell, Leclerc, Norris |
| Future Star Rookie | avg age < 23 | Antonelli, Hadjar, Lindblad, Bortoleto |
| Consistent Scorer | remaining | Piastri, Gasly, Hulkenberg, Ocon, Bearman |

---

## Anomaly Detection

Isolation Forest with `contamination=0.05` applied to:
`grid_position`, `finish_position`, `race_points`, `pit_stops`, `lap_time_gap_sec`

DNF finishing positions are replaced with 22 before fitting to avoid false anomalies from null values.

- Anomalies flagged: 26 (~5% of 528 race starts)
- Includes: back-of-grid wins, unexpected frontrunner DNFs, giant-killing drives

---

## GenAI RAG System

A TF-IDF retrieval-augmented generation pipeline built on 12 knowledge base documents covering:

- 2026 regulation changes (Active Aero, Overtake Mode, 50/50 power split, car dimensions)
- 2026 standings and race results (Australia, China Sprint/Qualifying)
- Team and driver profiles (Cadillac debut, Audi works entry, Hamilton to Ferrari, etc.)
- 2025 season summary (Norris champion by 2 points)

Usage:

```python
result = f1_rag_query("What is Overtake Mode in F1 2026?")
print(result['relevance_score'])  # cosine similarity
print(result['answer'])           # top-2 retrieved documents joined
```

Production upgrade path: replace TF-IDF with sentence-transformers BERT embeddings and add an LLM backend for generated answers rather than raw document retrieval.

---

## 2026 Season Context

The 2026 regulations are the most significant overhaul since 2022:

| Change | Detail |
|--------|--------|
| Active Aero | Moveable front and rear wings replace DRS |
| Overtake Mode | ERS boost available within 1 second of the car ahead |
| Power split | 50% ICE, 50% electric (up from ~20% electric in 2025) |
| MGU-H removed | Simplified hybrid, but MGU-K power up from 120 kW to 350 kW |
| Car weight | 768 kg (30 kg lighter than 2025) |
| Car dimensions | 20 cm shorter, 10 cm narrower |
| Fuel | 100% Advanced Sustainable Fuel |

Notable team changes for 2026:
- Lewis Hamilton moved from Mercedes to Ferrari
- Cadillac entered as the 11th constructor (Ferrari customer engine)
- Audi entered as a works team (formerly Sauber/Kick Sauber)
- Red Bull switched from Honda to their own RBPT-Ford power unit
- Alpine switched from Renault to Mercedes customer engine

---

## 2025 Season Summary

- Driver Champion: Lando Norris (McLaren) - 393 pts
- Constructor Champion: McLaren - 773 pts
- Runner-up driver: Max Verstappen (Red Bull) - 391 pts (margin: 2 points)
- Most race wins: Norris and Verstappen tied at 8 each; Piastri won 6
- McLaren won both titles for the first time since 2008

---

## Requirements

```
pandas >= 2.0
numpy
scikit-learn >= 1.3
matplotlib
seaborn
scipy
```

Install:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

---

## Quick Start

```bash
git clone https://github.com/osho-m/pitwall-analytics.git
cd pitwall-analytics
pip install pandas numpy scikit-learn matplotlib seaborn scipy
jupyter notebook F1_2026_DataScience_Project_v2.ipynb
```

All CSV files must be in the same directory as the notebook before running.

---

## Known Fixes Applied in v2

Two bugs from earlier versions are resolved in this notebook:

1. `GradientBoostingClassifier` TypeError — `class_weight` is not a valid constructor argument for GBC. Fixed by using `compute_sample_weight` and passing `sample_weight` to `.fit()`.

2. `ValueError` in Spearman correlation — mismatched array lengths caused by applying a DNF filter to one array but not the other. Fixed by using `df25.loc[no_dnf, col]` for both inputs.

---

## Dataset on Kaggle

The CSV files for this project are available as a Kaggle dataset. Each file includes a data card with column descriptions, suggested ML tasks, and methodology notes.

---

## License

MIT License. Free to use, modify, and distribute with attribution.

Data sources: race results and standings based on Formula One World Championship official data. 2026 regulation details based on FIA Technical Regulations. Performance ratings are analytical derivations, not official team data. Formula 1 and F1 are trademarks of Formula One Licensing BV.
