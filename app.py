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
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- UPDATED SECTION ---
# Load the new MULTI-CLASS model and its helper files
try:
    model = joblib.load('model_multiclass.pkl')
    scaler = joblib.load('scaler_multiclass.pkl')
    feature_names = joblib.load('feature_names_multiclass.pkl')
    print("Multi-class model and helper files loaded successfully.")
except FileNotFoundError:
    print("Error: Multi-class model files not found. Please run the updated training notebook first.")
    model, scaler, feature_names = None, None, None
# ---------------------

@app.route('/')
def home():
    if not feature_names:
        return "Error: Model not trained or files not found.", 500
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Error: Model not loaded.", 500

    try:
        data_dict = {feature: [float(request.form.get(feature))] for feature in feature_names}
        input_df = pd.DataFrame.from_dict(data_dict)
        input_scaled = scaler.transform(input_df)
        print("mode: ",model)
        print("inpute df: ",input_df)
        print("input scaled: ",input_scaled)
        # --- CHANGE: Logic for multi-class prediction ---
        prediction_num = model.predict(input_scaled)[0] # Get the predicted class (0, 1, or 2)
        prediction_proba = model.predict_proba(input_scaled)[0] # Get probabilities for all classes
        print("predict num: ",prediction_num)
        print("pridiction_proba: ",prediction_proba)
        # Define the class labels in the correct order
        class_labels = ['False Positive', 'Planetary Candidate', 'Confirmed Exoplanet']
        
        # Get the label and confidence for the predicted class
        result_text = class_labels[prediction_num]
        confidence = prediction_proba[prediction_num] * 100
        # -----------------------------------------------

        return render_template('results.html', 
                               prediction=result_text, 
                               confidence=f"{confidence:.2f}%")

    except (ValueError, TypeError):
        return "Error: Invalid input. Please ensure all fields are filled with numbers.", 400
    except Exception as e:
        return f"An unexpected error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)