from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json 

import shap 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

FEATURE_NAME_MAP = {
    'koi_score': 'Disposition Score',
    'koi_fpflag_nt': 'Not Transit-Like Flag',
    'koi_fpflag_ss': 'Stellar Eclipse Flag',
    'koi_fpflag_co': 'Centroid Offset Flag',
    'koi_period': 'Orbital Period',
    'koi_duration': 'Transit Duration',
    'koi_depth': 'Transit Depth',
    'koi_prad': 'Planet Radius',
    'koi_teq': 'Equilibrium Temperature',
    'koi_insol': 'Insolation Flux',
    'koi_model_snr': 'Signal-to-Noise Ratio',
    'koi_steff': 'Stellar Temperature'
}

try:
    model = joblib.load('model_multiclass.pkl')
    scaler = joblib.load('scaler_multiclass.pkl')
    feature_names = joblib.load('feature_names_multiclass.pkl')
    
    with open('famous_exoplanets.json', 'r') as f:
        FAMOUS_EXOPLANETS = json.load(f)
    explainer = shap.TreeExplainer(model)

except FileNotFoundError as e:
    model, scaler, feature_names, FAMOUS_EXOPLANETS, explainer = None, None, None, None, None

def create_light_curve(duration, depth):
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

        steff = data.get('koi_steff', 5800) 
        star_color = "#FFD700" 
        if steff <= 3700: star_color = "#FF6347"  
        elif steff <= 5200: star_color = "#FFA500" 
        elif steff <= 6000: star_color = "#FFD700" 
        elif steff <= 7500: star_color = "#FFFFE0" 
        elif steff > 7500: star_color = "#ADD8E6"  
            
        planet_size = data.get('koi_prad', 1)

        period = data.get('koi_period', 0)
        orbital_distance = 50 + 40 * np.log10(period + 1) if period > 0 else 50
        
        visualization_data = {
            "star_color": star_color,
            "planet_size": min(max(planet_size, 0.5), 15), 
            "orbital_distance": min(orbital_distance, 200) 
        }

        insolation = data.get('koi_insol', 0) 
        habitable_zone_status = "Unknown"
        
        if insolation > 1.1:
            habitable_zone_status = "Too Hot"
        elif insolation >= 0.35:
            habitable_zone_status = "Habitable Zone"
        elif insolation > 0:
            habitable_zone_status = "Too Cold"

        user_period = data.get('koi_period', 0)
        user_prad = data.get('koi_prad', 0)
        
        best_match = min(FAMOUS_EXOPLANETS, key=lambda x: abs(x['koi_period'] - user_period))
        
        comparison_text = (
            f"Its orbital period of {user_period:.1f} days is similar to {best_match['name']}, "
            f"which orbits its star in {best_match['koi_period']:.1f} days."
        )

        comparison_data = {
            "name": best_match['name'],
            "description": best_match['description'],
            "text": comparison_text
        }
        shap_values = explainer.shap_values(input_scaled)
        shap_values_for_prediction = None
        
        if isinstance(shap_values, list) and len(shap_values) == 3:
            shap_values_for_prediction = shap_values[prediction_num][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_for_prediction = shap_values[0, :, prediction_num]
        else:
            return jsonify({"error": "Unexpected SHAP value format."}), 500

        feature_shap_values = sorted(
            zip(feature_names, shap_values_for_prediction),
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        top_features = []
        for feature, shap_value in feature_shap_values[:4]:
            top_features.append({
                "feature": FEATURE_NAME_MAP.get(feature, feature),
                "value": data.get(feature, 'N/A'),
                "contribution": "positive" if shap_value > 0 else "negative"
            })
            
        explanation_data = {
            "predicted_class": result_text,
            "top_features": top_features
        }

        return jsonify({
            "prediction": result_text,
            "confidence": f"{confidence:.2f}%",
            "plot_url": plot_url,
            "visualization_data": visualization_data,
            "habitable_zone_status": habitable_zone_status,
            "comparison_data": comparison_data, 
            "explanation_data": explanation_data
        })

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)