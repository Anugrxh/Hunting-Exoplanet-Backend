Of course. Here is a complete and professional README.md file tailored specifically for your exoplanet prediction backend project.

Exoplanet Hunter AI Backend 🪐
<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python">
<img alt="Framework" src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask">
<img alt="Scikit-learn" src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img alt="API" src="https://img.shields.io/badge/API-REST-orange?style=for-the-badge">
<img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

This repository contains the backend server for the Exoplanet Hunter AI, a full-stack application designed to classify celestial objects from the NASA Kepler dataset. The API serves a trained machine learning model and provides rich, contextual data for a frontend client.

This project was developed for the 2025 NASA Space Apps Challenge.

✨ Key Features
Multi-Class Classification: Predicts if a Kepler Object of Interest (KOI) is a Confirmed Exoplanet, Planetary Candidate, or False Positive.

User-Friendly Model: Utilizes a RandomForestClassifier trained on a curated set of the 12 most predictive and intuitive features for ease of use.

Model Explainability: Integrates the SHAP library to explain why the model made a specific decision by identifying the most influential features for each prediction.

Rich Data Endpoints: The API provides a wealth of contextual data for each prediction, including:

A simulated light curve plot (as a base64 string).

Data for a 2D star system visualization.

Habitable zone status (Too Hot, Habitable, or Too Cold).

A comparison to a similar, famous real-world exoplanet.

Performance Metrics: A dedicated endpoint serves key performance metrics of the currently deployed model.

🛠️ Technology Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, SHAP, Pandas, NumPy

Plotting: Matplotlib

Production Server: Gunicorn

⚙️ Setup and Installation
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8+

pip and venv

1. Clone the Repository
Bash

git clone <your-repository-url>
cd <your-repository-folder>
2. Create and Activate a Virtual Environment
Windows:

Bash

python -m venv venv
.\venv\Scripts\activate
macOS / Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Add Model and Data Files
This repository does not store the large trained model files. You must generate them by running the provided Jupyter/Colab training notebook. After running the notebook, place the following downloaded files into the root of this project folder:

model_multiclass.pkl (the trained model)

scaler_multiclass.pkl (the data scaler)

feature_names_multiclass.pkl (the list of model features)

metrics.json (the model performance metrics)

famous_exoplanets.json (the comparison data)

▶️ Running the Application
To run the Flask development server for local testing:

Bash

python app.py
The API will be available at http://127.0.0.1:5000.

To run with the production server (Gunicorn):

Bash

gunicorn app:app
📄 API Endpoints
1. Prediction Endpoint
URL: /predict

Method: POST

Description: Classifies a single exoplanet candidate based on input features.

Request Body:
The body must be a JSON object containing the 12 required features.

Sample Request:

JSON

{
    "koi_score": 1.000,
    "koi_fpflag_nt": 0,
    "koi_fpflag_ss": 0,
    "koi_fpflag_co": 0,
    "koi_period": 3.522,
    "koi_duration": 3.19,
    "koi_depth": 807.9,
    "koi_prad": 2.83,
    "koi_teq": 1088,
    "koi_insol": 360.5,
    "koi_model_snr": 38.4,
    "koi_steff": 5805
}
Sample Success Response (200 OK):

JSON

{
    "comparison_data": {
        "description": "A 'Hot Jupiter' known for its deep blue color...",
        "name": "HD 189733b",
        "text": "Its orbital period of 3.5 days is similar to HD 189733b..."
    },
    "confidence": "98.00%",
    "explanation_data": {
        "predicted_class": "Planetary Candidate",
        "top_features": [
            {"contribution": "positive", "feature": "Koi Score", "value": 1.0},
            {"contribution": "positive", "feature": "Koi Model Snr", "value": 38.4},
            {"contribution": "negative", "feature": "Koi Fpflag Ss", "value": 0},
            {"contribution": "negative", "feature": "Koi Fpflag Co", "value": 0}
        ]
    },
    "habitable_zone_status": "Too Hot",
    "plot_url": "iVBORw0KGgoAAAANSUhEUgAABAAAA...",
    "prediction": "Planetary Candidate",
    "visualization_data": {
        "orbital_distance": 84.2,
        "planet_size": 2.83,
        "star_color": "#FFD700"
    }
}
2. Metrics Endpoint
URL: /metrics

Method: GET

Description: Returns the key performance metrics of the trained model.

Sample Success Response (200 OK):

JSON

{
    "accuracy": 0.985,
    "candidate_recall": 0.95,
    "confirmed_f1_score": 0.97,
    "false_positive_precision": 0.99
}
