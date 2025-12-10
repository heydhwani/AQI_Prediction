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

# AQI TREND OVER TIME


plt.figure(figsize=(12,4))
plt.plot(df["Datetime"], df["AQI"], color="red", linewidth=0.7)
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.grid(alpha=0.3)
plt.show()

# PM2.5 DISTRIBUTION


plt.figure(figsize=(7,4))
plt.hist(df["PM2.5"], bins=40, color="skyblue")
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.show()

#  PM10 DISTRIBUTION


plt.figure(figsize=(7,4))
plt.hist(df["PM10"], bins=40, color="orange")
plt.title("PM10 Distribution")
plt.xlabel("PM10")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.show()