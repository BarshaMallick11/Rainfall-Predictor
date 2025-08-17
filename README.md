# 🌧️ Rainfall Prediction for Smart Cities using Machine Learning

## 📌 Project Overview
This project focuses on predicting rainfall for **15 major cities of India (2010–2024)** using weather parameters such as temperature, humidity, wind speed, and precipitation.  
The goal is to support **smart city planning** by forecasting rainfall patterns and identifying potential **flood risks**.

---

## 🔬 Methodology
1. **Data Collection** – Weather dataset (2010–2024) for 15 Indian cities.  
2. **Data Preprocessing** – Cleaning, handling missing values, feature selection.  
3. **Model Training** – Applied:
   - Linear Regression  
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Random Forest Regressor  
4. **Model Evaluation** – Performance measured using metrics:  
   - MAE, MSE, RMSE  
   - MSLE, MAPE, SMAPE  
5. **Best Model Selection** – Random Forest performed best among all models.  
6. **Deployment** – Web application built using **Flask (Python)** with **HTML, CSS, and JavaScript** frontend.

---

## 📊 Results
- Random Forest achieved the **highest accuracy** and lowest error values.  
- The model successfully predicts rainfall amount based on weather conditions.  
- Provides insights for **flood management** and **urban planning**.

---

## 💻 Web Application
The project includes a **Flask-based web app**:
- Input: City, Year, Weather Parameters (auto/manual).  
- Output: Predicted Rainfall & Flood Risk Level.  
- User-friendly interface built with **HTML, CSS, JavaScript**.  

---

## 📂 Project Structure
Rainfall-Prediction/
│-- dataset/ # Weather dataset (2010–2024)
│-- models/ # Trained ML models
│-- app.py # Flask backend
│-- static/ # CSS, JS files
│-- templates/ # HTML files
│-- requirements.txt # Dependencies
│-- README.md # Project documentation


---

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rainfall-prediction.git
   cd rainfall-prediction
2. Install dependencies:
     pip install -r requirements.txt
3. Run the Flask app:
     python app.py
4. Open in browser:
     http://127.0.0.1:5000


🚀 Future Work

Integrate deep learning models (LSTM, GRU) for better accuracy.

Expand dataset to more cities and real-time weather APIs.

Deploy web app on cloud platforms (Heroku, AWS, Azure).

Add mobile-friendly UI for accessibility.
