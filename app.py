from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model & scalers
final_model = pickle.load(open('model/Parkinson_Model.pkl', 'rb'))
scaler_X = pickle.load(open('model/scaler.pkl', 'rb'))
scaler_y = pickle.load(open('model/scaler_y.pkl', 'rb'))

@app.route('/')
def index():
    return jsonify({"message": "Parkinson's Risk Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        input_data = np.array([[ 
            float(data['DATSCAN_PUTAMEN_R']),
            float(data['DATSCAN_PUTAMEN_L']),
            float(data['DATSCAN_CAUDATE_R']),
            float(data['DATSCAN_CAUDATE_L']),
            float(data['NP3TOT']),
            float(data['UPSIT_PRCNTGE']),
            float(data['COGCHG'])
        ]])
        input_data_scaled = scaler_X.transform(input_data)
        pred_scaled = final_model.predict(input_data_scaled)[0]
        risk_percent = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"
        return jsonify({
            "Parkinson_Risk_Percentage": round(risk_percent, 2),
            "Risk_Status": risk_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
