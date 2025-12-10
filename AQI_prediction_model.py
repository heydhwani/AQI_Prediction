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

# LABEL ENCODING (City, Station)

le_city = LabelEncoder()
le_station = LabelEncoder()

df["City_encoded"] = le_city.fit_transform(df["City"])
df["Station_encoded"] = le_station.fit_transform(df["Station"])

# SELECT FEATURES
feature_cols = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "City_encoded", "Station_encoded"
]

X = df[feature_cols]
y = df["AQI"]     # Target column

# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN XGBOOST MODEL

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)