# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import numpy as np

# # Initialize the Flask application
# app = Flask(__name__)

# # Load the trained model, scaler, and feature names
# try:
#     model = joblib.load('exoplanet_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     feature_names = joblib.load('feature_names.pkl')
#     print("Model, scaler, and feature names loaded successfully.")
# except FileNotFoundError:
#     print("Error: Model files not found. Please run train_model.py first.")
#     model, scaler, feature_names = None, None, None

# # Define the route for the homepage
# @app.route('/')
# def home():
#     if not feature_names:
#         return "Error: Model not trained. Please run the training script first.", 500
#     return render_template('index.html', features=feature_names)

# # Define the route to handle the prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if not model or not scaler or not feature_names:
#         return "Error: Model not loaded. Cannot make predictions.", 500
#     print(model)
#     print("-----#####------")
#     print(scaler)
#     print("-----#####------")
#     print(feature_names)
#     print("-----#####------")
#     try:
#         # Get the input data from the form
#         # `request.form.get` is used to safely access form data
#         # The `float` conversion handles numerical input
#         input_data = [float(request.form.get(feature)) for feature in feature_names]
#         print("-----#####------")
#         print(input_data)
#         print("-----#####------")
#         # Convert the list to a 2D numpy array, as the scaler expects it
#         input_array = np.array(input_data).reshape(1, -1)
#         print("-----#####------")
#         print(input_array)
#         # Scale the input data using the loaded scaler
#         input_scaled = scaler.transform(input_array)
        
#         # Make the prediction
#         prediction = model.predict(input_scaled)
#         prediction_proba = model.predict_proba(input_scaled)
        
#         # Determine the result and confidence
#         if prediction[0] == 1:
#             result_text = "Likely an Exoplanet"
#             confidence = prediction_proba[0][1] * 100
#         else:
#             result_text = "Likely NOT an Exoplanet"
#             confidence = prediction_proba[0][0] * 100

#         # Render the results page with the prediction outcome
#         return render_template('results.html', 
#                                prediction=result_text, 
#                                confidence=f"{confidence:.2f}%")

#     except Exception as e:
#         # Handle potential errors, like non-numerical input
#         return f"An error occurred: {e}", 400

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

#============REFINED CODE BY REMOVING UNWANTED FIELDS==========================================
# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import numpy as np

# # Initialize the Flask application
# app = Flask(__name__)

# # --- UPDATED SECTION ---
# # Load the new user-friendly model and its helper files
# try:
#     model = joblib.load('exoplanet_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     feature_names = joblib.load('feature_names.pkl')
#     print("User-friendly model and helper files loaded successfully.")
# except FileNotFoundError:
#     print("Error: Model files not found. Please run your training notebook and save the new model files first.")
#     model, scaler, feature_names = None, None, None
# # ---------------------

# @app.route('/')
# def home():
#     if not feature_names:
#         return "Error: Model not trained or files not found. Please run the training script first.", 500
#     # The index.html template will automatically update because it loops through the feature_names list
#     return render_template('index.html', features=feature_names)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if not model or not scaler or not feature_names:
#         return "Error: Model not loaded. Cannot make predictions.", 500

#     try:
#         # Create a dictionary from the form data
#         data_dict = {feature: [float(request.form.get(feature))] for feature in feature_names}
        
#         # Convert the dictionary to a pandas DataFrame to ensure correct feature order and naming
#         input_df = pd.DataFrame.from_dict(data_dict)
        
#         # Scale the DataFrame's values
#         input_scaled = scaler.transform(input_df)
        
#         # Make a prediction
#         prediction = model.predict(input_scaled)
#         prediction_proba = model.predict_proba(input_scaled)
        
#         # Determine the result and confidence score
#         if prediction[0] == 1:
#             result_text = "Likely a Confirmed Exoplanet"
#             confidence = prediction_proba[0][1] * 100
#         else:
#             result_text = "Likely a False Positive"
#             confidence = prediction_proba[0][0] * 100

#         return render_template('results.html', 
#                                prediction=result_text, 
#                                confidence=f"{confidence:.2f}%")

#     except (ValueError, TypeError):
#         return "Error: Invalid input. Please ensure all fields are filled with numbers.", 400
#     except Exception as e:
#         return f"An unexpected error occurred: {e}", 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

#==========UPDATED CODE WITH MULTI CLASS CLASSIFICATION================
# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import numpy as np
# import json 
# import matplotlib.pyplot as plt
# import numpy as np
# import io
# import base64

# app = Flask(__name__)

# def create_light_curve(duration, depth):
#     """Generates a simplified, simulated light curve plot."""
#     try:
#         # Create a time series
#         time = np.linspace(-10, 10, 500)
#         flux = np.ones_like(time)
        
#         # Create the transit dip
#         transit_start = -duration / 2
#         transit_end = duration / 2
#         flux[(time > transit_start) & (time < transit_end)] -= depth / 1e6 # Depth is in parts per million

#         # Add a bit of noise
#         flux += np.random.normal(0, 0.00005, size=flux.shape)

#         # Create the plot
#         fig, ax = plt.subplots(figsize=(8, 4), facecolor='#161b22')
#         ax.plot(time, flux, '.', color='#79c0ff', markersize=3)
#         ax.set_facecolor('#161b22')
#         ax.set_xlabel("Time from Transit Center (Hours)", color='white')
#         ax.set_ylabel("Relative Brightness (Flux)", color='white')
#         ax.tick_params(axis='x', colors='white')
#         ax.tick_params(axis='y', colors='white')
#         for spine in ax.spines.values():
#             spine.set_edgecolor('white')

#         # Save plot to a memory buffer
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', transparent=True)
#         buf.seek(0)
        
#         # Encode buffer to a base64 string to embed in HTML
#         plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
#         plt.close(fig)
#         return plot_url
#     except Exception:
#         return None
# # --- UPDATED SECTION ---
# # Load the new MULTI-CLASS model and its helper files
# try:
#     model = joblib.load('model_multiclass.pkl')
#     scaler = joblib.load('scaler_multiclass.pkl')
#     feature_names = joblib.load('feature_names_multiclass.pkl')
#     print("Multi-class model and helper files loaded successfully.")
# except FileNotFoundError:
#     print("Error: Multi-class model files not found. Please run the updated training notebook first.")
#     model, scaler, feature_names = None, None, None
# # ---------------------

# @app.route('/')
# def home():
#     if not feature_names:
#         return "Error: Model not trained or files not found.", 500
#     try:
#         with open('metrics.json', 'r') as f:
#             metrics = json.load(f)
#     except FileNotFoundError:
#         metrics = None # Handle case where file might be missing
    
#     return render_template('index.html', features=feature_names, metrics=metrics)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if not model:
#         return "Error: Model not loaded.", 500

#     try:
#         data_dict = {feature: [float(request.form.get(feature))] for feature in feature_names}
#         input_df = pd.DataFrame.from_dict(data_dict)
#         input_scaled = scaler.transform(input_df)
#         print("mode: ",model)
#         print("inpute df: ",input_df)
#         print("input scaled: ",input_scaled)
#         # --- CHANGE: Logic for multi-class prediction ---
#         prediction_num = model.predict(input_scaled)[0] # Get the predicted class (0, 1, or 2)
#         prediction_proba = model.predict_proba(input_scaled)[0] # Get probabilities for all classes
#         print("predict num: ",prediction_num)
#         print("pridiction_proba: ",prediction_proba)
#         # Define the class labels in the correct order
#         class_labels = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
        
#         # Get the label and confidence for the predicted class
#         result_text = class_labels[prediction_num]
#         confidence = prediction_proba[prediction_num] * 100
#         # -----------------------------------------------

#         return render_template('results.html', 
#                                prediction=result_text, 
#                                confidence=f"{confidence:.2f}%")

#     except (ValueError, TypeError):
#         return "Error: Invalid input. Please ensure all fields are filled with numbers.", 400
#     except Exception as e:
#         return f"An unexpected error occurred: {e}", 500

# if __name__ == '__main__':
#     app.run(debug=True)

##==========MATRIX FOR DATA VISUALIZATION============

# app.py

# --- Core Flask and ML Imports ---
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import json

# --- Imports for Stretch Goals (Plotting) ---
import matplotlib
matplotlib.use('Agg') # A non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Helper Files ---
try:
    # Load the multi-class model and its corresponding helper files
    model = joblib.load('model_multiclass.pkl')
    scaler = joblib.load('scaler_multiclass.pkl')
    feature_names = joblib.load('feature_names_multiclass.pkl')
    print("Multi-class model and helper files loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run the training notebook and save the model files first.")
    model, scaler, feature_names = None, None, None

# --- Helper Function for Stretch Goal 1 (Light Curve Plot) ---
def create_light_curve(duration, depth):
    """Generates a simplified, simulated light curve plot based on user input."""
    try:
        # Create a time series for the x-axis
        time = np.linspace(-10, 10, 500)
        # Start with a flat flux (brightness) of 1
        flux = np.ones_like(time)
        
        # Calculate the start and end of the transit based on its duration
        transit_start = -duration / 2
        transit_end = duration / 2
        # Create the dip in brightness. Depth is in parts per million, so we divide by 1e6.
        flux[(time > transit_start) & (time < transit_end)] -= depth / 1e6

        # Add a small amount of random noise to make it look more realistic
        flux += np.random.normal(0, 0.00005, size=flux.shape)

        # Create the plot with a dark theme
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#161b22')
        ax.plot(time, flux, '.', color='#79c0ff', markersize=3)
        ax.set_facecolor('#161b22')
        ax.set_xlabel("Time from Transit Center (Hours)", color='white')
        ax.set_ylabel("Relative Brightness (Flux)", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Save the plot to a memory buffer instead of a file
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        
        # Encode the image in the buffer to a base64 string to embed directly in HTML
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig) # Close the figure to free up memory
        return plot_url
    except Exception:
        # If anything goes wrong, just return nothing
        return None

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the homepage with the input form and model metrics."""
    if not feature_names:
        return "Error: Model not trained or files not found.", 500
    
    # Stretch Goal 2: Load metrics for the homepage
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None # Handle case where file might be missing
    
    return render_template('index.html', features=feature_names, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, makes a prediction, and renders the results page."""
    if not model:
        return "Error: Model not loaded.", 500

    try:
        # Get data from the form and convert to a DataFrame
        data_dict = {feature: [float(request.form.get(feature))] for feature in feature_names}
        input_df = pd.DataFrame.from_dict(data_dict)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make a multi-class prediction
        prediction_num = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Get the human-readable label and confidence score
        class_labels = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
        result_text = class_labels[prediction_num]
        confidence = prediction_proba[prediction_num] * 100
        
        # --- Stretch Goal 1: Generate the light curve plot ---
        duration = float(request.form.get('koi_duration'))
        depth = float(request.form.get('koi_depth'))
        plot_url = create_light_curve(duration, depth)
        
        # Pass all data to the results template
        return render_template('results.html', 
                               prediction=result_text, 
                               confidence=f"{confidence:.2f}%",
                               confidence_raw=confidence, # For Stretch Goal 3 slider
                               plot_url=plot_url)

    except (ValueError, TypeError):
        return "Error: Invalid input. Please ensure all fields are filled with valid numbers.", 400
    except Exception as e:
        return f"An unexpected error occurred: {e}", 500

# --- Main execution block ---
if __name__ == '__main__':
    app.run(debug=True)