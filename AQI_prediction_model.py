import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib


# LOAD CLEANED DATASET

df = pd.read_csv("Dataset/station_hour_clean.csv")

# If bucket column still exists, drop it
if "AQI_Bucket" in df.columns:
    df.drop(columns=["AQI_Bucket"], inplace=True)

# Convert Datetime to datetime type
df["Datetime"] = pd.to_datetime(df["Datetime"])
