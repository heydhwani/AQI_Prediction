import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'D:\Projects\AQI_Prediction\Dataset\station_hour.csv')
df = df.drop(columns=['AQI_Bucket'])

df['Datetime'] = pd.to_datetime(df['Datetime'])


