
# Exoplanet Hunter Backend 🪐

<p align="center">
  <img src="https://ichef.bbci.co.uk/images/ic/480xn/p0m6nj30.jpg.webp" alt="Exoplanet Hunter API Banner" width="100%">
</p>

# 🪐 Exoplanet Hunter API
> A machine learning API built with Flask to classify celestial objects from the NASA Kepler dataset.


A machine learning API built with Flask to classify celestial objects from the NASA Kepler dataset. This project serves as the backend for the Exoplanet Hunter application, providing predictions and rich, contextual data for frontend visualizations.

#### This project was developed as part of the 2025 NASA Space Apps Challenge.

## Key Features

- #### Multi-Class Classification: 
  Predicts if a   Kepler Object of Interest (KOI) is   a Confirmed Exoplanet, Planetary Candidate, or   False Positive.
- #### User-Friendly Model:
  Utilizes a RandomForestClassifier trained on a curated set of the 12 most predictive and intuitive features.
- #### Rich Data Endpoints:
  The API provides a wealth of contextual data for each prediction, including:

      - Simulated light curve plots.

      - Data for a 2D star system visualization.

      - Habitable zone status (Too Hot, Habitable, Too Cold).

      - A comparison to a similar, famous real-world exoplanet.
- Cross platform
## Tech Stack

**Backend:** Python, Flask

**Machine Learning:** Scikit-learn, Pandas, NumPy

**Plotting:** Matplotlib

**Server:** Gunicorn (for production)


## ⚙️ Setup and Installation
    
Follow these steps to set up and run the project on your local machine.

### Prerequisites
- Python 3.8+
- pip and venv

  - #### Clone the Repository
        git clone <your-repository-url>
        cd <your-repository-folder>
  -  Create and Activate a Virtual         Environment
      - #### Windows:
            python -m venv venv
            .\venv\Scripts\activate
      - #### macOS / Linux:
            python3 -m venv venv
            source venv/bin/activate
      - #### Install Dependencies
            pip install -r requirements.txt

### ▶️ Running the Application
   #### To run the Flask development server for local testing:
        python app.py
        




## API Reference

#### Prediction Endpoint

```http
  POST /predict
```


####  Metrics Endpoint

```http
  GET /metrics
```



## Deployment

To deploy this project run

```bash
  npm i gunicorn
  pip freeze > requirements.txt
  pip install -r requirements.txt
  gunicorn app:app
```


## 🔗 👨‍💻 Collaborators

- #### Anugrah M V
  [![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://anugrah-m-v.netlify.app/)

  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anugrah-m-v-187203271/)

- ### Mishab 
  [![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/mishab339/)

  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammed-mishab-p-b1497b293/)
