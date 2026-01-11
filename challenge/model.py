from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle


class DelayModel:

    # Top 10 features identified in the notebook
    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    @staticmethod
    def _get_period_day(date_str: str) -> str:
        """
        Get period of day from date string.

        Args:
            date_str (str): Date string in format 'YYYY-MM-DD HH:MM:SS'

        Returns:
            str: Period of day ('mañana', 'tarde', 'noche')
        """
        try:
            date_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").time()

            # Define time ranges
            morning_start = datetime.strptime("05:00", "%H:%M").time()
            morning_end = datetime.strptime("11:59", "%H:%M").time()
            afternoon_start = datetime.strptime("12:00", "%H:%M").time()
            afternoon_end = datetime.strptime("18:59", "%H:%M").time()
            evening_start = datetime.strptime("19:00", "%H:%M").time()
            evening_end = datetime.strptime("23:59", "%H:%M").time()
            night_start = datetime.strptime("00:00", "%H:%M").time()
            night_end = datetime.strptime("04:59", "%H:%M").time()

            if morning_start <= date_time <= morning_end:
                return "mañana"
            elif afternoon_start <= date_time <= afternoon_end:
                return "tarde"
            elif evening_start <= date_time <= evening_end:
                return "noche"
            elif night_start <= date_time <= night_end:
                return "noche"
            else:
                return "noche"  # Default to night for edge cases
        except (ValueError, AttributeError):
            return "noche"  # Default for invalid dates

    @staticmethod
    def _is_high_season(fecha: str) -> int:
        """
        Determine if date is in high season.

        Args:
            fecha (str): Date string in format 'YYYY-MM-DD HH:MM:SS'

        Returns:
            int: 1 if high season, 0 otherwise
        """
        try:
            fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
            year = fecha_dt.year

            # Define high season periods
            periods = [
                # Dec 15 - Dec 31
                (datetime(year, 12, 15), datetime(year, 12, 31)),
                # Jan 1 - Mar 3
                (datetime(year, 1, 1), datetime(year, 3, 3)),
                # Jul 15 - Jul 31
                (datetime(year, 7, 15), datetime(year, 7, 31)),
                # Sep 11 - Sep 30
                (datetime(year, 9, 11), datetime(year, 9, 30)),
            ]

            # Check if date falls within any high season period
            for start, end in periods:
                if start <= fecha_dt <= end:
                    return 1

            return 0
        except (ValueError, AttributeError):
            return 0  # Default to not high season for invalid dates

    @staticmethod
    def _get_min_diff(row) -> float:
        """
        Calculate difference in minutes between scheduled and actual departure.

        Args:
            row: DataFrame row with Fecha-O and Fecha-I columns

        Returns:
            float: Difference in minutes
        """
        try:
            fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
            fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
            min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
            return min_diff
        except (ValueError, KeyError, AttributeError):
            return 0.0  # Default to no difference for invalid data

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Make a copy to avoid modifying original data
        data_processed = data.copy()

        # Feature engineering
        if "Fecha-I" in data_processed.columns:
            data_processed["period_day"] = data_processed["Fecha-I"].apply(
                self._get_period_day
            )
            data_processed["high_season"] = data_processed["Fecha-I"].apply(
                self._is_high_season
            )

        if "Fecha-O" in data_processed.columns and "Fecha-I" in data_processed.columns:
            data_processed["min_diff"] = data_processed.apply(
                self._get_min_diff, axis=1
            )
            data_processed["delay"] = np.where(data_processed["min_diff"] > 15, 1, 0)

        # Create dummy variables for categorical features
        features = pd.DataFrame(index=data_processed.index)

        # Handle OPERA column
        if "OPERA" in data_processed.columns:
            opera_dummies = pd.get_dummies(data_processed["OPERA"], prefix="OPERA")
            features = pd.concat([features, opera_dummies], axis=1)

        # Handle TIPOVUELO column
        if "TIPOVUELO" in data_processed.columns:
            tipovuelo_dummies = pd.get_dummies(
                data_processed["TIPOVUELO"], prefix="TIPOVUELO"
            )
            features = pd.concat([features, tipovuelo_dummies], axis=1)

        # Handle MES column
        if "MES" in data_processed.columns:
            mes_dummies = pd.get_dummies(data_processed["MES"], prefix="MES")
            features = pd.concat([features, mes_dummies], axis=1)

        # Ensure all expected top features are present (fill with 0 if missing)
        for col in self.TOP_FEATURES:
            if col not in features.columns:
                features[col] = 0

        # Select only the top features in the correct order
        features = features[self.TOP_FEATURES]

        if target_column and target_column in data_processed.columns:
            target = data_processed[[target_column]]
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Shuffle the data
        features_shuffled, target_shuffled = shuffle(features, target, random_state=111)

        # Calculate class weights for balancing
        n_y0 = len(target_shuffled[target_shuffled.iloc[:, 0] == 0])
        n_y1 = len(target_shuffled[target_shuffled.iloc[:, 0] == 1])
        scale_pos_weight = n_y0 / n_y1 if n_y1 > 0 else 1

        # Train XGBoost model with class balancing
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale_pos_weight
        )

        self._model.fit(features_shuffled, target_shuffled.values.ravel())

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        predictions_proba = self._model.predict_proba(features)
        # Convert probabilities to binary predictions (threshold at 0.5)
        predictions = [1 if pred[1] > 0.5 else 0 for pred in predictions_proba]

        return predictions
