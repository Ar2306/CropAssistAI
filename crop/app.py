# File: app.py
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# IMPORTANT: If you used joblib to save your models:
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # change in production

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------
# Load Models / Scalers / LabelEncoders
# ---------------------------

# Crop model
crop_model = joblib.load(os.path.join(MODELS_DIR, 'crop_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODELS_DIR, 'crop_scaler.pkl'))
crop_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'crop_label.pkl'))

# Fertilizer model
fertilizer_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
fertilizer_scaler = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_scaler.pkl'))
fertilizer_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_label.pkl'))

# Plant disease model (Keras .h5 example)
# Only load if needed:
try:
    from tensorflow.keras.models import load_model
    plant_disease_model = load_model(os.path.join(MODELS_DIR, 'Plant_Detection_Model.h5'))
except Exception as e:
    plant_disease_model = None
    print("Plant disease model not loaded:", e)

# ---------------------------
# Prediction Functions
# ---------------------------

def predict_crop(input_data):
    """
    input_data: list -> [N, P, K, temperature, humidity, ph, rainfall]
    """
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = pd.DataFrame([input_data], columns=columns)
    scaled = crop_scaler.transform(df)
    pred = crop_model.predict(scaled)
    crop_name = crop_label_encoder.inverse_transform(pred)[0]
    return crop_name

@app.route('/fertilizer-prediction', methods=['GET', 'POST'])
def fertilizer_prediction():
    # Mapping dictionaries to convert dropdown selections to numeric codes
    soil_type_mapping = {
        'Sandy': 0,
        'Loamy': 1,
        'Black': 2,
        'Red': 3,
        'Clayey': 4
    }
    crop_type_mapping = {
        'Maize': 0,
        'Sugarcane': 1,
        'Cotton': 2,
        'Tobacco': 3,
        'Paddy': 4
    }
    
    if request.method == 'POST':
        try:
            # Get dropdown selections as strings and convert to numbers using the mappings
            soil_str = request.form['Soil Type']
            crop_str = request.form['Crop Type']
            soil_numeric = soil_type_mapping.get(soil_str)
            crop_numeric = crop_type_mapping.get(crop_str)
            if soil_numeric is None or crop_numeric is None:
                flash("Invalid selection for Soil Type or Crop Type.")
                return redirect(url_for('fertilizer_prediction'))
            
            input_data = {
                'Temparature': float(request.form['Temparature']),
                'Humidity': float(request.form['Humidity']),
                'Moisture': float(request.form['Moisture']),
                'Soil Type': soil_numeric,
                'Crop Type': crop_numeric,
                'Nitrogen': float(request.form['Nitrogen']),
                'Potassium': float(request.form['Potassium']),
                'Phosphorous': float(request.form['Phosphorous'])
            }
            result = predict_fertilizer_numeric(input_data)
            return render_template('fertilizer_prediction.html', prediction=result)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('fertilizer_prediction'))
    return render_template('fertilizer_prediction.html')
plant_disease_class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy', 
    'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]


plant_disease_info = {
    'Apple___Apple_scab': {
         'description': "A fungal disease causing dark, scabby lesions on apple leaves and fruit.",
         'recommended_fertilizer': "Use a balanced fertilizer combined with fungicide treatments."
    },
    'Apple___Black_rot': {
         'description': "Causes decay and black lesions on apple fruit and leaves.",
         'recommended_fertilizer': "Apply high-potassium fertilizer and ensure proper pruning."
    },
    'Apple___Cedar_apple_rust': {
         'description': "A rust disease that affects apple trees in association with juniper trees.",
         'recommended_fertilizer': "Use a balanced fertilizer and practice cultural methods to reduce humidity."
    },
    'Apple___healthy': {
         'description': "The apple tree is healthy with no visible disease symptoms.",
         'recommended_fertilizer': "Maintain balanced nutrition for optimal growth."
    },
    'Blueberry___healthy': {
         'description': "The blueberry plant is healthy.",
         'recommended_fertilizer': "Use an acidic fertilizer (e.g., ammonium sulfate) suitable for blueberries."
    },
    'Cherry___healthy': {
         'description': "The cherry tree is healthy.",
         'recommended_fertilizer': "Apply a balanced fertilizer with essential micronutrients."
    },
    'Cherry___Powdery_mildew': {
         'description': "A fungal infection causing a white, powdery coating on leaves.",
         'recommended_fertilizer': "Use balanced fertilizer and consider fungicides to control the disease."
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
         'description': "Fungal disease causing gray spots on corn leaves.",
         'recommended_fertilizer': "Apply nitrogen-rich fertilizer and follow proper crop rotation."
    },
    'Corn___Common_rust': {
         'description': "Rust disease marked by reddish-brown pustules on corn leaves.",
         'recommended_fertilizer': "Maintain balanced fertilizer and select resistant corn varieties."
    },
    'Corn___Northern_Leaf_Blight': {
         'description': "A fungal disease causing elongated lesions on corn leaves.",
         'recommended_fertilizer': "Use balanced fertilizer with an emphasis on potassium."
    },
    'Corn___healthy': {
         'description': "The corn plant is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition based on soil testing."
    },
    'Grape___Black_rot': {
         'description': "A fungal disease that causes dark, decaying spots on grape clusters.",
         'recommended_fertilizer': "Apply a balanced fertilizer and ensure proper canopy management."
    },
    'Grape___Esca_(Black_Measles)': {
         'description': "A complex disease causing black measles on grapevines leading to decline.",
         'recommended_fertilizer': "Use a phosphorus-rich fertilizer and manage irrigation properly."
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
         'description': "A leaf spot disease causing defoliation in grapevines.",
         'recommended_fertilizer': "Apply balanced fertilizer and fungicides if needed."
    },
    'Grape___healthy': {
         'description': "The grapevine is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition."
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
         'description': "A severe bacterial disease causing yellowing, stunted growth, and fruit drop in citrus.",
         'recommended_fertilizer': "Use specialized citrus fertilizer with added micronutrients."
    },
    'Peach___Bacterial_spot': {
         'description': "Bacterial spot causes dark spots on peach leaves and fruit.",
         'recommended_fertilizer': "Apply balanced fertilizer and consider bactericides."
    },
    'Peach___healthy': {
         'description': "The peach tree is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition."
    },
    'Pepper,_bell___Bacterial_spot': {
         'description': "Bacterial spot on bell peppers causes lesions on leaves and fruit.",
         'recommended_fertilizer': "Use balanced fertilizer along with copper-based bactericides."
    },
    'Pepper,_bell___healthy': {
         'description': "The bell pepper plant is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition."
    },
    'Potato___Early_blight': {
         'description': "A fungal disease that causes dark spots and defoliation on potato leaves.",
         'recommended_fertilizer': "Use potassium-rich fertilizer and apply fungicides early."
    },
    'Potato___Late_blight': {
         'description': "A severe disease that rapidly decays potato foliage and tubers.",
         'recommended_fertilizer': "Apply balanced fertilizer and aggressive fungicide treatment."
    },
    'Potato___healthy': {
         'description': "The potato plant is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition based on soil tests."
    },
    'Raspberry___healthy': {
         'description': "The raspberry plant is healthy.",
         'recommended_fertilizer': "Apply balanced fertilizer with adequate potassium."
    },
    'Soybean___healthy': {
         'description': "The soybean plant is healthy.",
         'recommended_fertilizer': "Use balanced fertilizer; soybeans may benefit from phosphorus enrichment."
    },
    'Squash___Powdery_mildew': {
         'description': "Powdery mildew causes a white, powdery coating on squash leaves.",
         'recommended_fertilizer': "Use balanced fertilizer and appropriate fungicides."
    },
    'Strawberry___Leaf_scorch': {
         'description': "Leaf scorch causes browning at the edges of strawberry leaves.",
         'recommended_fertilizer': "Apply organic fertilizer and manage irrigation carefully."
    },
    'Strawberry___healthy': {
         'description': "The strawberry plant is healthy.",
         'recommended_fertilizer': "Maintain balanced, preferably organic, nutrition."
    },
    'Tomato___Bacterial_spot': {
         'description': "Bacterial spot causes dark lesions on tomato leaves and fruit.",
         'recommended_fertilizer': "Use balanced fertilizer with copper-based bactericides."
    },
    'Tomato___Early_blight': {
         'description': "Early blight causes concentric lesions on tomato leaves.",
         'recommended_fertilizer': "Apply balanced fertilizer with additional potassium."
    },
    'Tomato___Late_blight': {
         'description': "Late blight is a devastating disease that rapidly decays tomato foliage.",
         'recommended_fertilizer': "Use balanced fertilizer and aggressive fungicide applications."
    },
    'Tomato___Leaf_Mold': {
         'description': "Leaf mold causes a moldy appearance on the underside of tomato leaves.",
         'recommended_fertilizer': "Improve air circulation, and use balanced fertilizer."
    },
    'Tomato___Septoria_leaf_spot': {
         'description': "Septoria leaf spot causes small, dark spots on tomato leaves.",
         'recommended_fertilizer': "Maintain balanced fertilizer and proper watering practices."
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
         'description': "Spider mite infestations cause stippling and yellowing on tomato leaves.",
         'recommended_fertilizer': "Maintain plant health with balanced fertilizer and consider miticides."
    },
    'Tomato___Target_Spot': {
         'description': "Target spot produces target-like lesions on tomato leaves.",
         'recommended_fertilizer': "Use balanced fertilizer and appropriate fungicide treatments."
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
         'description': "This virus causes yellowing and curling of tomato leaves.",
         'recommended_fertilizer': "Focus on overall plant health; manage the virus through cultural practices."
    },
    'Tomato___Tomato_mosaic_virus': {
         'description': "Tomato mosaic virus causes mosaic patterns and leaf distortion on tomatoes.",
         'recommended_fertilizer': "Maintain balanced nutrition and remove severely infected plants."
    },
    'Tomato___healthy': {
         'description': "The tomato plant is healthy.",
         'recommended_fertilizer': "Maintain balanced nutrition."
    }
}

def predict_plant_disease(image_path):
    if not plant_disease_model:
        return "No plant disease model loaded."
    try:
        # Open and preprocess the image
        image_obj = Image.open(image_path).convert('RGB')
        image_obj = image_obj.resize((224, 224))  # Adjust size as needed
        img_array = np.array(image_obj) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        preds = plant_disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        
        # Map the numeric prediction to the disease name using the class names list
        disease_name = plant_disease_class_names[class_idx]
        return disease_name
    except Exception as e:
        return f"Error in disease prediction: {str(e)}"
# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')  # or an index.html that extends base.html

@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            data = [N, P, K, temperature, humidity, ph, rainfall]
            result = predict_crop(data)
            return render_template('crop_prediction.html', prediction=result)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('crop_prediction'))
    return render_template('crop_prediction.html')


# ---------------------------
# Prediction Function
# ---------------------------
def predict_fertilizer_numeric(input_data):
    """
    Predicts the fertilizer based on input agricultural parameters (with numeric categorical values).

    Args:
        input_data (dict): Keys include:
            'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous'
    Returns:
        str: Predicted fertilizer name.
    """
    df = pd.DataFrame([input_data])
    # Scale only the numerical columns (the categorical ones are already numeric)
    num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    df[num_cols] = fertilizer_scaler.transform(df[num_cols])
    pred = fertilizer_model.predict(df)
    
    # Try using inverse_transform if available.
    if hasattr(fertilizer_label_encoder, 'inverse_transform'):
        fertilizer_name = fertilizer_label_encoder.inverse_transform(pred)[0]
    else:
        # Create a mapping dictionary using the unique fertilizer names
        fertilizer_mapping = {
            0: 'Urea',
            1: 'DAP',
            2: '14-35-14',
            3: '28-28',
            4: '17-17-17',
            5: '20-20',
            6: '10-26-26'
        }
        fertilizer_name = fertilizer_mapping.get(pred[0], "Unknown Fertilizer")
    return fertilizer_name


@app.route('/plant-disease-detection', methods=['GET', 'POST'])
def plant_disease_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # Save and predict
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        disease = predict_plant_disease(file_path)
        return render_template('plant_disease.html',
                               prediction=disease,
                               image_url=url_for('static', filename=f'uploads/{filename}'))
    return render_template('plant_disease.html')

if __name__ == '__main__':
    app.run(debug=True)
