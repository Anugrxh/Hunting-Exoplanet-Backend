from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model_multiclass.pkl')
    scaler = joblib.load('scaler_multiclass.pkl')
    feature_names = joblib.load('feature_names_multiclass.pkl')
    print("Multi-class model and helper files loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run the training notebook and save the model files first.")
    model, scaler, feature_names = None, None, None


def create_light_curve(duration, depth):
    """Generates a simplified, simulated light curve plot based on user input."""
    try:
        time = np.linspace(-10, 10, 500)
        flux = np.ones_like(time)
        
        transit_start = -duration / 2
        transit_end = duration / 2
        flux[(time > transit_start) & (time < transit_end)] -= depth / 1e6

        flux += np.random.normal(0, 0.00005, size=flux.shape)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#161b22')
        ax.plot(time, flux, '.', color='#79c0ff', markersize=3)
        ax.set_facecolor('#161b22')
        ax.set_xlabel("Time from Transit Center (Hours)", color='white')
        ax.set_ylabel("Relative Brightness (Flux)", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig) 
        return plot_url
    except Exception:
        return None


@app.route('/')
def home():
    return "<h1>Exoplanet Prediction API</h1><p>Send a POST request to /predict</p>"

@app.route('/metrics')
def get_metrics():
    """Reads the metrics file and returns it as JSON."""
    try:
        with open('metrics.json', 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    except FileNotFoundError:
        return jsonify({"error": "Metrics file not found. Please train the model first."}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        input_df = pd.DataFrame([data])
        input_df = input_df[feature_names]
        
        input_scaled = scaler.transform(input_df)
        
        prediction_num = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        class_labels = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
        result_text = class_labels[prediction_num]
        confidence = prediction_proba[prediction_num] * 100
        duration = data.get('koi_duration', 0)
        depth = data.get('koi_depth', 0)
        plot_url = create_light_curve(duration, depth)

        return jsonify({
            "prediction": result_text,
            "confidence": f"{confidence:.2f}%",
            "plot_url":plot_url
        })

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)