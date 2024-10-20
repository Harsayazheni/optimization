import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('energy-consumption.csv')

def categorize_consumption(consumption):
    if consumption < 200:
        return 'Very Low'
    elif 200 <= consumption < 400:
        return 'Low'
    elif 400 <= consumption < 600:
        return 'Medium'
    else:
        return 'High'

data['Consumption Category'] = data['energy_consumption'].apply(categorize_consumption)

features = ['temperature', 'humidity', 'hour', 'day', 'month']
target = 'energy_consumption'

X = data[features]
y = data[target]

# Get min and max values for temperature and humidity
temp_min, temp_max = X['temperature'].min(), X['temperature'].max()
humidity_min, humidity_max = X['humidity'].min(), X['humidity'].max()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model (you might want to do this separately and save the model)
def train_model():
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'energy_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    return best_model

# Load or train the model
try:
    model = joblib.load('energy_model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    model = train_model()
    scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html', temp_min=temp_min, temp_max=temp_max, 
                           humidity_min=humidity_min, humidity_max=humidity_max)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = data['temperature']
    humidity = data['humidity']
    hour = data['hour']
    day = data['day']
    month = data['month']
    
    # Generate predictions for each month
    monthly_data = []
    for m in range(1, 13):
        input_data = [temperature, humidity, hour, day, m]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        monthly_data.append(prediction)
    
    # Use the prediction for the current month
    current_prediction = monthly_data[month - 1]
    category = categorize_consumption(current_prediction)
    
    return jsonify({
        'predicted_consumption': current_prediction,
        'category': category,
        'monthly_data': monthly_data
    })

if __name__ == '__main__':
    app.run(debug=True)