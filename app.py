from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model & scalers
MODEL_PATH = "model/Parkinson_Model.pkl"
SCALER_X_PATH = "model/scaler.pkl"
SCALER_Y_PATH = "model/scaler_y.pkl"

# Check if model and scalers exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
    raise FileNotFoundError("Model or scaler files are missing. Ensure all required files are in the 'model/' directory.")

final_model = pickle.load(open(MODEL_PATH, 'rb'))
scaler_X = pickle.load(open(SCALER_X_PATH, 'rb'))
scaler_y = pickle.load(open(SCALER_Y_PATH, 'rb'))

@app.route('/')
def index():
    return jsonify({"message": "Parkinson's Risk Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Validate required keys
        required_keys = ['DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 'DATSCAN_CAUDATE_R',
                         'DATSCAN_CAUDATE_L', 'NP3TOT', 'UPSIT_PRCNTGE', 'COGCHG']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required key: {key}"}), 400

        # Convert input to NumPy array
        input_data = np.array([[ 
            float(data['DATSCAN_PUTAMEN_R']),
            float(data['DATSCAN_PUTAMEN_L']),
            float(data['DATSCAN_CAUDATE_R']),
            float(data['DATSCAN_CAUDATE_L']),
            float(data['NP3TOT']),
            float(data['UPSIT_PRCNTGE']),
            float(data['COGCHG'])
        ]])

        # Scale the input using the trained scaler
        input_data_scaled = scaler_X.transform(input_data)

        # Make prediction
        pred_scaled = final_model.predict(input_data_scaled)

        # Convert scaled prediction back to original scale
        risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Risk Status Classification
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        return jsonify({
            "Parkinson_Risk_Percentage": round(risk_percent, 2),
            "Risk_Status": risk_status
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if _name_ == '__main__':
    app.run(debug=True)
