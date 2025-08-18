import os
from flask import Flask, request, jsonify, render_template, abort
import pandas as pd
import numpy as np
from flask_cors import CORS
import joblib

# Initialize the Flask application
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- DATA LOADING ---
# We will use the CSV file as our data source for predictions.
DATA_PATH = "RandomForest_Hybrid_Model_Predictions.csv"

try:
    # Load the entire CSV into a pandas DataFrame
    predictions_df = pd.read_csv(DATA_PATH)
    # Clean up city names for consistency
    predictions_df['City'] = predictions_df['City'].str.strip().str.title()
    VALID_CITIES = sorted(predictions_df['City'].unique().tolist())
    print(f"Data loaded successfully from {DATA_PATH}. Found {len(VALID_CITIES)} cities.")
except FileNotFoundError:
    print(f"FATAL ERROR: Data file not found at {DATA_PATH}. The application cannot run.")
    predictions_df = pd.DataFrame() # Make it an empty DataFrame
    VALID_CITIES = []
except Exception as e:
    print(f"FATAL ERROR: Could not read or process data file: {e}")
    predictions_df = pd.DataFrame()
    VALID_CITIES = []
# --- NEW: LOAD MODELS (FOR AUTOMATIC WEATHER PREDICTION) ---
try:
    weather_model = joblib.load('weather_model.pkl')
    weather_model_columns = joblib.load('weather_model_columns.pkl')
    weather_target_columns = joblib.load('weather_target_columns.pkl')
    city_encoder = joblib.load('city_encoder.pkl')
    month_encoder = joblib.load('month_encoder.pkl') 
    # Note: We don't need rainfall_model.pkl because the final prediction uses the CSV
    print("ML Models for automatic weather prediction loaded successfully.")
except FileNotFoundError:
    print("WARNING: One or more .pkl files for the automatic feature are missing. The automatic section may not work.")
    weather_model = None
except Exception as e:
    print(f"ERROR loading ML models: {e}")
    weather_model = None

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serves the main frontend page (index.html)."""
    return render_template('index.html')

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """API endpoint to provide the list of valid cities to the frontend."""
    if not VALID_CITIES:
        return jsonify({"error": "City data is not available on the server."}), 500
    return jsonify(VALID_CITIES)

# NEW: Route for Automatic Weather Parameter Prediction
@app.route('/predict_weather_auto', methods=['POST'])
def predict_weather_auto():
    """
    Predicts weather parameters using the ML model based on city, year, and month.
    This endpoint is ONLY for the 'Automatically' button.
    """
    if not weather_model:
        return jsonify({"error": "Weather prediction model is not loaded on the server."}), 500
        
    try:
        data = request.get_json()
        year = int(data['year'])
        # The month is received as a string name, we need to handle it.
        # The training.py did not save a month_encoder, so we assume a mapping.
        # If you have a month_encoder.pkl, we can load and use it.
        # For now, let's create a simple mapping based on standard month order.
        # The month is received as a text name (e.g., "January")
        month_name = data['month']
        city = data['city'].title() # Ensure title case to match encoder
        
        # FIXED: Use the loaded month_encoder to transform the month name into the encoded value the model expects.
        month_encoded = month_encoder.transform(np.array([month_name]))[0]
        city_encoded = city_encoder.transform(np.array([city]))[0]
        
        # Prepare input DataFrame for the weather prediction model
        weather_input = pd.DataFrame(columns=weather_model_columns)
        weather_input.loc[0] = 0
        weather_input['year'] = year
        weather_input['City_encoded'] = city_encoded
        # FIXED: Use the correct column name 'month_encoded' that the model was trained on.
        weather_input['month_encoded'] = month_encoded
        
        # Predict future weather conditions
        predicted_values = weather_model.predict(weather_input[weather_model_columns])[0]

        # Create a dictionary with the results
        predicted_weather = dict(zip(weather_target_columns, predicted_values))

        return jsonify(predicted_weather)

    except Exception as e:
        return jsonify({"error": f"An internal server error occurred during weather prediction: {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend.
    This function now finds the closest match in the CSV file instead of using a .pkl model.
    """
    if predictions_df.empty:
        abort(500, description="Prediction data is not loaded on the server.")

    try:
        data = request.get_json()
        city = data.get("city")
        
        # --- Input Validation ---
        if not city or city not in VALID_CITIES:
            return jsonify({"error": f"Unsupported or missing city."}), 400

        # --- Find Closest Match (Simulation) ---
        # Filter the dataframe for the selected city
        city_df = predictions_df[predictions_df['City'] == city].copy()
        if city_df.empty:
            return jsonify({"error": f"No data available in the CSV for {city}."}), 404

        # Define the parameters to compare
        params_to_compare = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
            'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m'
        ]
        
        # Get user inputs for these parameters
        user_inputs = pd.Series({param: float(data[param]) for param in params_to_compare})

        # Calculate the Euclidean distance between the user's input and every row in the city's data
        # This finds the most similar historical weather condition in the CSV
        distances = np.sqrt(((city_df[params_to_compare] - user_inputs)**2).sum(axis=1))
        
        # Get the index of the row with the smallest distance
        closest_match_index = distances.idxmin()
        
        # Retrieve the entire row for the closest match
        closest_match = predictions_df.loc[closest_match_index]
        
        # Get the pre-calculated predicted rainfall from that row
        predicted_precip = closest_match['Predicted_Rainfall']
        
        display_rain = round(float(predicted_precip), 4)

        # --- Flood Risk Classification ---
        if predicted_precip <= 0.5:
            risk = "Low"
        elif predicted_precip <= 0.7:
            risk = "Moderate"
        else:
            risk = "High"

        # --- Response ---
        return jsonify({
            "City": city,
            "Predicted_Rainfall_mm": display_rain,
            "Flood_Risk_Level": risk
        })

    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input data: {e}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=False)

