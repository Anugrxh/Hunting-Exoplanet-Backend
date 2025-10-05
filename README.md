# 🪐 Exoplanet Hunter API

A **machine learning API** built with **Flask** to classify celestial objects from the **NASA Kepler dataset**.  
This project serves as the **backend for the Exoplanet Hunter application**, providing real-time predictions, visualizations, and contextual astronomy data.

> 🚀 Developed as part of the **2025 NASA Space Apps Challenge**

---

## 🌟 Key Features

- **🔭 Multi-Class Classification:**  
  Predicts whether a Kepler Object of Interest (KOI) is a  
  - Confirmed Exoplanet  
  - Planetary Candidate  
  - False Positive  

- **🧠 Smart Model:**  
  Uses a `RandomForestClassifier` trained on **12 highly predictive and interpretable features**.

- **💬 Explainable AI:**  
  Integrated with **SHAP** to explain each prediction by identifying the most influential features.

- **📊 Rich Data Endpoints:**  
  Each prediction includes:
  - Simulated **light curve plot**  
  - **2D star-system visualization data**  
  - **Habitable zone status** (Too Hot / Habitable / Too Cold)  
  - **Comparison** with a real-world exoplanet of similar nature  

- **📈 Model Metrics Endpoint:**  
  Provides **accuracy**, **precision**, **recall**, and **F1 scores** for all prediction classes.

---

## 🧰 Technology Stack

| Category | Technologies |
|-----------|---------------|
| **Backend** | Flask (Python) |
| **Machine Learning** | Scikit-learn, SHAP, Pandas, NumPy |
| **Plotting** | Matplotlib |
| **Server (Prod)** | Gunicorn |

---

## ⚙️ Setup & Installation

### ✅ Prerequisites
- Python **3.8+**
- `pip` and `venv`

---

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
