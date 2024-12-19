from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
log_reg_model = joblib.load('log_reg_model.joblib')
best_rf_model = joblib.load('best_rf_model.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Convert form data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # One-hot encoding (ensure same columns as training)
    df_encoded = pd.get_dummies(df, drop_first=True)
    prediction_lr = log_reg_model.predict(df_encoded)[0]
    prediction_rf = best_rf_model.predict(df_encoded)[0]

    return jsonify({
        'Logistic Regression Prediction': 'High' if prediction_lr == 1 else 'Low',
        'Random Forest Prediction': 'High' if prediction_rf == 1 else 'Low'
    })


if __name__ == '__main__':
    app.run(debug=True)
