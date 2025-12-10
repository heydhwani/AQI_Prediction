import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv("Dataset/station_hour.csv")

# Drop bucket column if present
if "AQI_Bucket" in df.columns:
    df.drop(columns=["AQI_Bucket"], inplace=True)

# Convert Datetime
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# Sort by time
df = df.sort_values("Datetime")


print("\n===== Dataset Info =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isna().sum())

print("\n===== Statistical Summary =====")
print(df.describe())