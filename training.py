import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("Starting model training for MONTHLY predictions...")

# --- Step 1: Load and Expand Data from Seasonal to Monthly ---
df = pd.read_csv('RandomForest_Hybrid_Model_Predictions.csv')

# Define the mapping from season to months
season_to_month_map = {
    'Jan-Feb': ['January', 'February'],
    'Mar-May': ['March', 'April', 'May'],
    'Jun-Sep': ['June', 'July', 'August', 'September'],
    'Oct-Dec': ['October', 'November', 'December'],
    'Annual': [] # We will ignore the 'Annual' rows for monthly predictions
}

# Create a new list to hold the monthly data
monthly_data = []

print("Expanding seasonal data into monthly data...")
# Iterate over each row in the original dataframe
for index, row in df.iterrows():
    season = row['season']
    months_in_season = season_to_month_map.get(season, [])
    
    # For each month in the season, create a new row
    for month in months_in_season:
        new_row = row.to_dict()
        new_row['month'] = month
        monthly_data.append(new_row)

# Create a new DataFrame from the monthly data
df_monthly = pd.DataFrame(monthly_data)
df_monthly = df_monthly.drop(columns=['season']) # Drop the old 'season' column

print(f"Data expanded. Original rows: {len(df)}, New monthly rows: {len(df_monthly)}")

# --- Step 2: Prepare Data for Modeling ---

# Encode 'City' and 'month' columns to numbers
city_encoder = LabelEncoder()
month_encoder = LabelEncoder()

df_monthly['City_encoded'] = city_encoder.fit_transform(df_monthly['City'])
df_monthly['month_encoded'] = month_encoder.fit_transform(df_monthly['month'])

# --- Step 3: Define Features and Train Models ---

# These are the features that describe the date and location
prediction_inputs = ['year', 'City_encoded', 'month_encoded']

# These are ALL the weather conditions we want to predict
weather_targets = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m'
]

rainfall_target = 'Actual_Rainfall'

# Train a Multi-Output Weather Model
print("Training multi-output weather model on monthly data...")
X_weather = df_monthly[prediction_inputs]
y_weather = df_monthly[weather_targets]

weather_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
weather_model.fit(X_weather, y_weather)

# Train the Final Rainfall Model
print("Training final rainfall model on monthly data...")
X_rainfall = df_monthly[weather_targets]
y_rainfall = df_monthly[rainfall_target]

rainfall_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rainfall_model.fit(X_rainfall, y_rainfall)

# --- Step 4: Save Models and Helper Files ---
print("Saving models and helper files...")
joblib.dump(weather_model, 'weather_model.pkl')
joblib.dump(rainfall_model, 'rainfall_model.pkl')
joblib.dump(prediction_inputs, 'weather_model_columns.pkl')
joblib.dump(weather_targets, 'weather_target_columns.pkl')
joblib.dump(city_encoder, 'city_encoder.pkl')
joblib.dump(month_encoder, 'month_encoder.pkl') # Save the new month encoder
joblib.dump(list(city_encoder.classes_), 'cities.pkl')
joblib.dump(list(month_encoder.classes_), 'months.pkl') # Save the list of month names

print("\nMonthly training complete!")
print("You can now run the web application with 'python app.py'")