import time
import logging
from typing import Dict, Any
import datetime
import lightgbm as lgb
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self):
        self.model = None
        self.last_trained_at = None
        self._load_model()

    def _load_model(self):
        """
        Load the pre-trained LightGBM model from best_model.txt.
        """
        logger.info("INFO: Attempting to load LightGBM model...")
        try:
            self.model = lgb.Booster(model_file='best_model.txt')
            self.last_trained_at = str(datetime.datetime.now()).split(".")[0]
            logger.info(f"INFO: LightGBM model loaded successfully at {self.last_trained_at}")
        except Exception as e:
            logger.error(f"ERROR: Failed to load LightGBM model: {e}")
            raise

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using the LightGBM model.
        Input data must include features matching the training data (20 features).
        Returns predictions in the PredictionResponse schema.
        """
        logger.info(f"INFO: Making prediction with data: {data}")

        # Extract input features
        home_team_odds_avg = data.get("home_team_odds_avg")
        away_team_odds_avg = data.get("away_team_odds_avg")
        home_team = data.get("home_team", "Unknown")
        away_team = data.get("away_team", "Unknown")

        # Construct feature vector (20 features to match training)
        feature_names = [
            'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A',
            'p_b365_h', 'p_b365_d', 'p_b365_a', 'home_goals_scored',
            'home_goals_conceded', 'home_win_rate', 'away_goals_scored',
            'away_goals_conceded', 'away_win_rate', 'elo_home_pre',
            'elo_away_pre', 'h2h_home_wins_last5', 'h2h_draws_last5',
            'h2h_away_wins_last5', 'goal_difference'
        ]
        feature_values = {
            'home_team_goal': 0,  # Not known pre-match
            'away_team_goal': 0,  # Not known pre-match
            'B365H': home_team_odds_avg,
            'B365D': (home_team_odds_avg + away_team_odds_avg) / 2,
            'B365A': away_team_odds_avg,
            'p_b365_h': 1 / home_team_odds_avg if home_team_odds_avg else 0,
            'p_b365_d': 1 / ((home_team_odds_avg + away_team_odds_avg) / 2),
            'p_b365_a': 1 / away_team_odds_avg if away_team_odds_avg else 0,
            'home_goals_scored': 0,  # Placeholder
            'home_goals_conceded': 0,  # Placeholder
            'home_win_rate': 0.5,  # Placeholder
            'away_goals_scored': 0,  # Placeholder
            'away_goals_conceded': 0,  # Placeholder
            'away_win_rate': 0.5,  # Placeholder
            'elo_home_pre': 1500,  # Default Elo
            'elo_away_pre': 1500,  # Default Elo
            'h2h_home_wins_last5': 0,  # Placeholder
            'h2h_draws_last5': 0,  # Placeholder
            'h2h_away_wins_last5': 0,  # Placeholder
            'goal_difference': 0  # Placeholder (home_team_goal - away_team_goal)
        }
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        input_df = input_df.astype(np.float32)

        # Make prediction
        try:
            prob = self.model.predict(input_df)[0]
            winner = home_team if prob > 0.5 else away_team
            winner_confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
            winner_odds = home_team_odds_avg if prob > 0.5 else away_team_odds_avg

            return {
                "winner": winner,
                "winner_confidence_pct": round(winner_confidence, 1),
                "winner_best_bet_odds": round(winner_odds, 2),
                "over_under": None,
                "over_under_confidence_pct": None,
                "over_under_best_bet_odds": None,
                "spread": None,
                "spread_confidence_pct": None,
                "spread_best_bet_odds": None
            }
        except Exception as e:
            logger.error(f"ERROR: Prediction failed: {e}")
            raise
