import numpy as np 
import pandas as pd 
import shap 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from flask import Flask, request, jsonify 

# 1. MODEL TRAINING & EXPLAINABLE AI
data = load_diabetes() 
X = pd.DataFrame(data.data, columns=data.feature_names) 
y = data.target 

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 

model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 

# SHAP Analysis
explainer = shap.Explainer(model, X_train) 
shap_values = explainer(X_test, check_additivity=False) 

# Generate Plots
shap.summary_plot(shap_values, X_test) 
shap.plots.waterfall(shap_values[0]) 

# Save Interactive Force Plot
shap.initjs() 
force_plot = shap.plots.force(shap_values[0]) 
shap.save_html("shap_force_plot.html", force_plot) 
print("SHAP interactive visualization saved as shap_force_plot.html") 

# 2. FLASK DEPLOYMENT
app = Flask(__name__) 

@app.route('/') 
def home(): 
    return "ML Model API is running. Use /predict endpoint." 

@app.route('/predict', methods=['POST']) 
def predict(): 
    input_data = request.json 
    # Convert input list to numpy array and reshape for prediction
    features = np.array(input_data["data"]).reshape(1, -1) 
    prediction = model.predict(features) 

    return jsonify({ 
        "prediction": int(prediction[0]) 
    }) 

if __name__ == '__main__': 
    # use_reloader=False is used to prevent issues when running in notebooks
    app.run(debug=True, use_reloader=False)
