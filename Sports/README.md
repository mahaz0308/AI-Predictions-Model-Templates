# Soccer Match Outcome Predictor Model

## Overview
This is a binary classification model built using LightGBM to predict soccer (football) match outcomes. It predicts whether the home team wins (label: 1) or not (label: 0, which includes draws or away wins). The model was trained with Optuna for hyperparameter optimization to maximize the F1 score.

## Training Data
- **Sport and Leagues**: Soccer (football) from multiple European leagues, including:
  - Swiss Super League (league_id: 24558, primary focus with most data).
  - English Premier League (league_id: 1729).
  - Italian Serie A (league_id: 10257).
  - Spanish La Liga (league_id: 21518).
  - Others like Dutch Eredivisie (league_id: 13274) and Portuguese Primeira Liga (league_id: 19694).
- **Dataset**: Sourced from `features.csv` (~10,000+ rows of historical matches from seasons 2008/2009 to 2015/2016).
- **Features** (20 in total, as listed in `preprocessing.json`):
  - Match metadata: league_id, home_team_goal, away_team_goal.
  - Betting odds: B365H, B365D, B365A, and derived probabilities (p_b365_h, p_b365_d, p_b365_a).
  - Team stats: home_goals_scored, home_goals_conceded, home_win_rate, away_goals_scored, away_goals_conceded, away_win_rate.
  - Ratings: elo_home_pre, elo_away_pre.
  - Head-to-head: h2h_home_wins_last5, h2h_draws_last5, h2h_away_wins_last5.
- **Target**: The "result" column, remapped from original values (1: home win → 1; 0: draw or 2: away win → 0).
- **Preprocessing**:
  - Dropped irrelevant columns: date, season, home_team_api_id, away_team_api_id.
  - Filled missing numeric values with means.
  - Ensured float32 types.
  - Used sample weights for class balancing.
- **Training Process** (from `model.py`):
  - Split: 80/20 train/test with stratification.
  - Optimized with Optuna (40 trials) for params like num_leaves, max_depth, etc.
  - Final model trained on full train set for 200 rounds.
  - Evaluation: F1 score on test set; full report in console output during training.

## Deployment Considerations
- **Input Requirements**:
  - Data must match the schema in `features.csv` or `sample_test.csv`.
  - Use `preprocess_input` function from `predict.py` to prepare inputs: drops unused columns, reorders features per `preprocessing.json`, fills NaNs with means, and converts to float32.
  - Example input: A CSV or DataFrame with columns like league_id, B365H, elo_home_pre, etc. (no target needed for prediction).
- **Dependencies**:
  - Python 3.x.
  - Libraries: pandas, numpy, lightgbm, json (for loading preprocessing.json).
  - No additional installations needed beyond these—install via `pip install pandas numpy lightgbm`.
- **Usage**:
  - Load model: `model = lgb.Booster(model_file="best_model.txt")`.
  - Predict: Use `predict` function in `predict.py`—returns binary predictions (1/0) and probabilities.
  - Example: Run `python predict.py` with `sample_test.csv` to see outputs.
  - Testing: Use `test.py` on `features.csv` to generate `predictions.csv` and evaluate performance.
- **Output**:
  - Binary: 1 (home win) if probability > 0.5, else 0 (draw or away win).
  - Probabilities: Raw sigmoid outputs from the model.
- **Limitations and Considerations**:
  - Data is historical (up to 2016), so retrain with recent data for better accuracy on current matches.
  - Assumes features like Elo ratings and head-to-head are pre-computed— you'll need a feature engineering pipeline for live data.
  - Handle class imbalance in production (model uses weights, but monitor).
  - No real-time dependencies (e.g., no API calls), but for scalability, deploy in a container (e.g., Docker) or cloud service.
  - Performance: Check `test.py` output for classification report (e.g., precision, recall, F1).
  - If deploying to a server, ensure file paths for `best_model.txt` and `preprocessing.json` are correct.

## Files Included
- `model.py`: Training script.
- `predict.py`: Inference script.
- `test.py`: Evaluation script on full data.
- `best_model.txt`: Trained LightGBM model.
- `preprocessing.json`: Column info for input processing.
- `features.csv`: Full training dataset.
- `sample_test.csv`: Sample inputs for testing.
- `predictions.csv`: Example outputs from test.py.

For questions, contact me at [mahazabbasi070@gmail.com].