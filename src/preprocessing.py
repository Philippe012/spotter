import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import holidays
from geopy.distance import geodesic
from src.config import (
    HOLIDAY_COUNTRY, HOLIDAY_STATE, CATEGORICAL_FEATURES,
    TARGET_COLUMN, ID_COLUMN, PROCESSED_TRAIN, PROCESSED_VALIDATION
)
from src.utils import setup_logger, validate_dataframe

logger = setup_logger('preprocessing')


class DataPreprocessor:
    def __init__(self, train_data: Optional[pd.DataFrame] = None):
        self.us_holidays = holidays.US(years=range(2024, 2027))
        self.fitted = False
        self.categorical_columns = []
        self.numerical_columns = []
        
        if train_data is not None:
            self.fit(train_data)
            
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if ID_COLUMN in df.columns:
            df = df.drop_duplicates(subset=[ID_COLUMN], keep='first')
            logger.info(f"Removed duplicates, new shape: {df.shape}")
            
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['dayofyear'] = df['date'].dt.dayofyear
            
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            df['is_holiday'] = df['date'].dt.date.apply(
                lambda d: 1 if d in self.us_holidays else 0
            )
            
            
            if "equipment" in df.columns:
                df["equipment"] = df["equipment"].str.strip().str.title()
                
            for col in ["pickup", "delivery"]:
                if col in df.columns:
                    df[col] = df[col].str.strip().str.title()
            
            numeric_cols = ["distance", "weight", "market_index", 'quote_signal']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    
                    
            if TARGET_COLUMN in df.columns:
                df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
                
                invalid_target = df[TARGET_COLUMN].isnull()
                if invalid_target.any():
                    logger.warning(f"Removing {invalid_target.sum():,} rows with invalid target")
                    df = df[~invalid_target]
            
            # Handle missing values
            df = self._handle_missing_values(df)
            return df
        
        def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    if df[col].dtype == "object":
                        df[col] = df[col].fillna("Unknown")
                        logger.debug(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
                        
                    else:
                        if col == TARGET_COLUMN:
                            continue
                        elif col in ["distance", "weight"]:
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                            logger.debug(f"Filled {missing_count} missing values in '{col}' with median: {median_val:.1f}")
                        else:
                            df[col] = df[col].fillna(0)
                            logger.debug(f"Filled {missing_count} missing values in '{col}' with 0")
                            
                        return df
        
        def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if 'pickup' in df.columns and 'delivery' in df.columns:
                df['route_pair'] = df['pickup'] + ' -> ' + df['delivery']           
                
            if TARGET_COLUMN in df.columns and 'distance' in df.columns:
                df['rate_per_mile'] = df[TARGET_COLUMN] / df['distance']
                df['rate_per_mile'] = df['rate_per_mile'].clip(0.5, 10.0)
                
            if "weight" in df.columns and "distance" in df.columns:
                df["weight_per_mile"] = df["weight"] / df["distance"]
                df["weight_per_mile"] = df["weight_per_mile"].clip(0, 100)
            
            if "market_index" in df.columns and "quote_signal" in df.columns:
                df["market_quote_interaction"] = df["market_index"] * df["quote_signal"]
                df["market_quote_ratio"] = df["market_index"] / (df["quote_signal"] + 0.001)
                
            return df
        
    def create_location_features(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        lat_cols = ['pickup_lat', 'delivery_lat']
        lon_cols = ['pickup_lon', 'delivery_lon']
        
        if all(col in df.columns for col in lat_cols + lon_cols):
            for col in lat_cols + lon_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['pickup_region'] = df['pickup_lat'].apply(self._get_region)
            df['delivery_region'] = df['delivery_lat'].apply(self._get_region)
            df['same_region'] = (df['pickup_region'] == df['delivery_region']).astype(int)
            
            df['lat_lon_interaction'] = df['pickup_lat'] * df['pickup_lon']
            df['delivery_lat_lon'] = df['delivery_lat'] * df['delivery_lon']
            
            df['lat_diff'] = np.abs(df['pickup_lat'] - df['delivery_lat'])
            df['lon_diff'] = np.abs(df['pickup_lon'] - df['delivery_lon'])
        
        return df
    
    
    def _get_region(self, lat: float) -> str:
        if pd.isna(lat):
            return "Unkown"
        if lat > 44:
            return "North"
        elif lat > 40:
            return "Northeast"
        elif lat > 37:
            return "Midwest"
        elif lat > 34:
            return "West"
        elif lat > 30:
            return "South"
        else:
            return "Southeast"
    
    def fit(self, df: pd.DataFrame):
        self.categorical_columns = [col for col in df.columns 
                                    if col in CATEGORICAL_FEATURES if col in df.columns]
        
        # I added engineered categorical columns
        for col in ["router_pair", "pickup_region", "delivery_region"]:
            if col in df.columns:
                self.categorical_columns.append(col)
                
        self.fitted = True
        logger.info(f"Preprocessor fitted. Categorical columns: {len(self.categorical_columns)}")
        return self
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()
        df = self.clean_data(df)
        
        df = self.create_route_features(df)
        df = self.create_location_features(df)
        
        
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
            
        logger.info(f"Transformed data shape: {df.shape}")
        return df
    
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    
    def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.fit_transform(df)
        logger.info(f"Preprocessing complete. Final shape: {processed_df.shape}")
        return processed_df
    
    