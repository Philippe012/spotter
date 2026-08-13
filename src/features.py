import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Tuple

from src.utils import setup_logger
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, TARGET_COLUMN, ID_COLUMN

logger = setup_logger("features")


class FeatureEngineer:
    def __init__(self):
        self.categorical_features = CATEGORICAL_FEATURES
        self.numerical_features = NUMERICAL_FEATURES
        self.preprocessor = None
        self.fitted = False
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols_ = None
        self.route_stats = None
        self.equip_stats = None

    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if TARGET_COLUMN in df.columns:
            if "route_pair" in df.columns:
                self.route_stats = df.groupby("route_pair")[TARGET_COLUMN].agg(["mean","std","min","max","count"])
                df = df.merge(self.route_stats, left_on="route_pair", right_index=True, how="left")

            if "equipment" in df.columns:
                self.equip_stats = (
                    df.groupby("equipment")[TARGET_COLUMN]
                    .agg(["mean", "std", "min", "max", "count"])
                    .add_prefix("equip_")
                )
                df = df.merge(self.equip_stats, left_on="equipment", right_index=True, how="left")
        else:
            if self.route_stats is not None and "route_pair" in df.columns:
                df = df.merge(self.route_stats, left_on="route_pair", right_index=True, how="left")
                for col in self.route_stats.columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(self.route_stats[col].mean())

            if self.equip_stats is not None and "equipment" in df.columns:
                df = df.merge(self.equip_stats, left_on="equipment", right_index=True, how="left")
                for col in self.equip_stats.columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(self.equip_stats[col].mean())
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["day_of_week"] = df["date"].dt.dayofweek
            df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
            df["day_of_year"] = df["date"].dt.dayofyear
            df["quarter"] = df["date"].dt.quarter
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
            df["days_in_month"] = df["date"].dt.days_in_month
            df["days_until_month_end"] = df["days_in_month"] - df["day"]

            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            df["season"] = df["month"].apply(self._get_season)
            df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
            df["is_peak_season"] = df["month"].isin([9, 10, 11]).astype(int)
        return df

    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        lat_cols = ["pickup_lat", "delivery_lat"]
        lon_cols = ["pickup_lon", "delivery_lon"]

        if all(col in df.columns for col in lat_cols + lon_cols):
            for col in lat_cols + lon_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["lat_diff"] = np.abs(df["pickup_lat"] - df["delivery_lat"])
            df["lon_diff"] = np.abs(df["pickup_lon"] - df["delivery_lon"])

            df["pickup_region"] = df["pickup_lat"].apply(self._get_region)
            df["delivery_region"] = df["delivery_lat"].apply(self._get_region)
            df["same_region"] = (df["pickup_region"] == df["delivery_region"]).astype(int)

            df["route_length_to_gap"] = df["distance"] / (df["lat_diff"] + df["lon_diff"] + 1e-6)
            df["distance_ratio_to_geo"] = df["distance"] / (self._haversine_miles_series(df) + 1e-6)
            df["midpoint_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2.0
            df["midpoint_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2.0

        return df

    def _haversine_miles_series(self, df: pd.DataFrame) -> pd.Series:
        def hav(lat1, lon1, lat2, lon2):
            if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                return 0.0
            radius = 3958.8
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)
            a = (
                np.sin(delta_phi / 2.0) ** 2
                + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return float(radius * c)

        return df.apply(
            lambda row: hav(row["pickup_lat"], row["pickup_lon"], row["delivery_lat"], row["delivery_lon"]),
            axis=1,
        )

    def _get_region(self, lat: float) -> str:
        if pd.isna(lat):
            return "Unknown"
        if lat >= 44:
            return "North"
        elif lat >= 39:
            return "Midwest"
        elif lat >= 35:
            return "Central"
        elif lat >= 30:
            return "South"
        return "Southeast"

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "distance" in df.columns and "weight" in df.columns:
            df["distance_weight"] = df["distance"] * df["weight"]
            df["weight_per_mile"] = df["weight"] / (df["distance"].replace(0, np.nan) + 1e-6)
            df["distance_per_weight"] = df["distance"] / (df["weight"].replace(0, np.nan) + 1e-6)

        if "market_index" in df.columns and "quote_signal" in df.columns:
            df["market_signal_strength"] = df["market_index"] / (df["quote_signal"].replace(0, np.nan) + 1e-6)
            df["market_quote_gap"] = df["market_index"] - df["quote_signal"]
            df["market_quote_ratio"] = df["market_index"] / (df["quote_signal"].replace(0, np.nan) + 1e-6)

        if "market_index" in df.columns:
            if "distance" in df.columns:
                df["market_distance"] = df["market_index"] * df["distance"]
            if "weight" in df.columns:
                df["market_weight"] = df["market_index"] * df["weight"]

        if "quote_signal" in df.columns:
            if "distance" in df.columns:
                df["quote_distance"] = df["quote_signal"] * df["distance"]
            if "weight" in df.columns:
                df["quote_weight"] = df["quote_signal"] * df["weight"]

        if "market_index" in df.columns and "quote_signal" in df.columns:
            df["market_quote"] = df["market_index"] * df["quote_signal"]
            df["market_quote_diff"] = df["market_index"] - df["quote_signal"]

        return df

    def create_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "weight" in df.columns:
            df["weight_class"] = pd.cut(
                df["weight"],
                bins=[0, 10000, 20000, 30000, 40000, 50000],
                labels=["Light", "Light-Medium", "Medium", "Medium-Heavy", "Heavy"],
            )

        if "distance" in df.columns:
            df["distance_class"] = pd.cut(
                df["distance"],
                bins=[0, 250, 500, 1000, 2000, 5000],
                labels=["Very Short", "Short", "Medium", "Long", "Very Long"],
            )

        if "equipment" in df.columns and "distance" in df.columns and "distance_class" in df.columns:
            df["equipment_distance"] = df["equipment"].astype(str) + "_" + df["distance_class"].astype(str)

        if "equipment" in df.columns and "weight" in df.columns:
            df["equipment_weight_band"] = df["equipment"].astype(str) + "_" + df["weight_class"].astype(str)

        if "route_pair" in df.columns:
            df["route_frequency"] = df["route_pair"].map(df["route_pair"].value_counts())

        return df

    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            vals = df[col].astype(str).fillna("Unknown")

            if self.fitted:
                encoder = self.label_encoders.get(col)
                if encoder is None:
                    raise RuntimeError(f"Encoder for {col} not found")
                known = set(encoder.classes_)
                mapped = vals.apply(lambda x: x if x in known else "Unknown")
                df[col] = mapped
                df[f"{col}_encoded"] = encoder.transform(mapped.values)
            else:
                uniques = pd.Series(vals.unique()).astype(str)
                if "Unknown" not in set(uniques):
                    uniques = pd.concat([uniques, pd.Series(["Unknown"])], ignore_index=True)
                encoder = LabelEncoder()
                encoder.fit(uniques.values)
                df[f"{col}_encoded"] = encoder.transform(vals.values)
                self.label_encoders[col] = encoder

        return df

    def scale_numerical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        available_cols = [
            col for col in columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not available_cols:
            return df

        if not self.fitted:
            self.numerical_cols_ = available_cols.copy()
            scaled_data = self.scaler.fit_transform(df[self.numerical_cols_].values)
        else:
            expected = self.numerical_cols_ or []
            X = np.zeros((len(df), len(expected)), dtype=float)
            for i, col in enumerate(expected):
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    X[:, i] = df[col].astype(float).values
                else:
                    X[:, i] = 0.0
            scaled_data = self.scaler.transform(X)

        for i, col in enumerate(self.numerical_cols_ if self.numerical_cols_ is not None else available_cols):
            df[f"{col}_scaled"] = scaled_data[:, i]

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()
        logger.info("Starting feature engineering...")

        df = self.create_temporal_features(df)
        df = self.create_geographic_features(df)
        df = self.create_aggregated_features(df)
        df = self.create_interaction_features(df)
        df = self.create_rate_features(df)

        categorical_cols = [
            "equipment",
            "pickup",
            "delivery",
            "route_pair",
            "pickup_region",
            "delivery_region",
            "season",
            "weight_class",
            "distance_class",
            "equipment_distance",
            "equipment_weight_band",
        ]
        available_categorical = [col for col in categorical_cols if col in df.columns]
        df = self.encode_categorical(df, available_categorical)

        numerical_cols = [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "weight_per_mile",
            "distance_per_weight",
            "lat_diff",
            "lon_diff",
            "route_length_to_gap",
            "distance_ratio_to_geo",
            "market_signal_strength",
            "market_quote_gap",
            "market_quote_ratio",
            "distance_weight",
            "market_distance",
            "market_weight",
            "quote_distance",
            "quote_weight",
            "market_quote",
            "route_frequency",
            "days_until_month_end",
        ]
        available_numerical = [col for col in numerical_cols if col in df.columns]
        df = self.scale_numerical(df, available_numerical)

        self.fitted = True

        exclude_cols = [ID_COLUMN, TARGET_COLUMN, "date"] + CATEGORICAL_FEATURES
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return df, feature_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform.")

        df = df.copy()
        df = self.create_temporal_features(df)
        df = self.create_geographic_features(df)
        df = self.create_aggregated_features(df)
        df = self.create_interaction_features(df)
        df = self.create_rate_features(df)

        categorical_cols = [
            "equipment",
            "pickup",
            "delivery",
            "route_pair",
            "pickup_region",
            "delivery_region",
            "season",
            "weight_class",
            "distance_class",
            "equipment_distance",
            "equipment_weight_band",
        ]
        available_categorical = [col for col in categorical_cols if col in df.columns]
        df = self.encode_categorical(df, available_categorical)

        numerical_cols = [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "weight_per_mile",
            "distance_per_weight",
            "lat_diff",
            "lon_diff",
            "route_length_to_gap",
            "distance_ratio_to_geo",
            "market_signal_strength",
            "market_quote_gap",
            "market_quote_ratio",
            "distance_weight",
            "market_distance",
            "market_weight",
            "quote_distance",
            "quote_weight",
            "market_quote",
            "route_frequency",
            "days_until_month_end",
        ]
        available_numerical = [col for col in numerical_cols if col in df.columns]
        df = self.scale_numerical(df, available_numerical)
        return df