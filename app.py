from flask import Flask, render_template, jsonify
import joblib
import os
# Ensure mvp_core.py is in the same directory and contains the required functions
from mvp_core import record_audio, preprocess_audio, extract_live_features, FS 
import warnings

# Suppress warnings from scikit-learn/joblib during loading
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Load Model & Scaler ---
MODEL_NAME = 'parkinsons_model.pkl'
SCALER_NAME = 'feature_scaler.pkl'

try:
    model = joblib.load(MODEL_NAME)
    scaler = joblib.load(SCALER_NAME)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load {MODEL_NAME} or {SCALER_NAME}. Did you run train_model.py? Details: {e}")
    model, scaler = None, None

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the HTML interface
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model files not loaded. Check terminal for training script instructions.'}), 500

    try:
        # 1. RECORD & PREPROCESS (Calls functions from mvp_core.py)
        raw_audio, current_sr = record_audio(duration=5, fs=FS)
        processed_audio = preprocess_audio(raw_audio, current_sr)
        
        if processed_audio.size == 0:
            return jsonify({
                'label': 0, 
                'score': "0.00%", 
                'status': 'No viable speech detected (try speaking louder).', 
                'risk_category': 'None'
            })

        # 2. FEATURE EXTRACTION
        live_features = extract_live_features(processed_audio, sr=FS)
        
        # 3. SCALING & PREDICTION
        scaled_features = scaler.transform(live_features)
        
        prediction_label = model.predict(scaled_features)[0] # Binary label (0 or 1)
        prediction_proba = model.predict_proba(scaled_features)[0][1] # Probability of class 1
        
        # --- NEW RISK RANGE LOGIC ---
        
        risk_category = "Low Risk"
        status = "Healthy Voice Signature"
        
        if prediction_proba >= 0.90:
            # High certainty of Parkinson's
            risk_category = "Critical Risk"
            status = "High Risk Detected (Urgent Follow-up Recommended)"
        elif prediction_proba >= 0.75:
            # Strong indication of risk
            risk_category = "High Risk"
            status = "Elevated Risk Detected"
        elif prediction_proba >= 0.50:
            # Above chance, warrants attention
            risk_category = "Moderate Risk"
            status = "Moderate Risk Detected"
        
        # 4. Return results, including the risk category for UI styling
        return jsonify({
            'label': int(prediction_label),
            'score': f"{prediction_proba*100:.2f}%",
            'status': status,
            'risk_category': risk_category
        })
        
    except Exception as e:
        # Catches unexpected errors in the extraction pipeline
        return jsonify({
            'error': f"Prediction failed due to: {str(e)}", 
            'label': 0, 
            'risk_category': 'Error',
            'status': 'Error: Check terminal'
        }), 500


if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Flask server...")
    # Use host='0.0.0.0' if you want to access it from your phone on the same network
    app.run(host='0.0.0.0', debug=True)