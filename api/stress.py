from flask import Blueprint, request, jsonify, make_response
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

stress_api = Blueprint("stress_api", __name__,
                        url_prefix='/api/stress')
api = Api(stress_api)
# Load the dataset
data = pd.read_csv('/kaggle/input/stress-level-detection/Stress-Lysis.csv')

# Split the data into features and labels
X = data.drop('Probability of Having Stress', axis=1)
y = data['Probability of Having Stress']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Function to predict the chance of being depressed
def predict_stress(humidity, temperature, step_count, stress_level):
    input_data = scaler.transform([[humidity, stress_level, step_count, stress_level]])
    chance_of_stress = model.predict(input_data)[0]
    return chance_of_stress

# Take user input
class Predict(Resource):
    def post(self):
        body=request.get_json()
        humidity = float(body.get("humidity"))
        temperature = float(body.get("temperature"))
        step_count = float(body.get("step count"))
        stress_level = float(body.get("sleep"))
        chance_of_stress = predict_stress(humidity, temperature, step_count, stress_level)
        return (jsonify(f"Based on the provided data, the chance of developing stress is: {chance_of_stress * 100:.2f}%"))

api.add_resource(Predict, '/')