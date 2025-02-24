import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
from dotenv import load_dotenv
load_dotenv()

########################
#   Flask App Setup
########################
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# Store uploaded images in static/uploads for direct serving
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

########################
#   Load ML/DL Models
########################
# Crop Recommendation Model
crop_model = joblib.load(os.path.join(MODELS_DIR, 'crop_model.pkl'))
crop_scaler = joblib.load(os.path.join(MODELS_DIR, 'crop_scaler.pkl'))
crop_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'crop_label.pkl'))

# Fertilizer Model
fertilizer_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
fertilizer_scaler = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_scaler.pkl'))
fertilizer_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'fertilizer_label.pkl'))

# Plant Disease Model (Keras)
try:
    from tensorflow.keras.models import load_model
    plant_disease_model = load_model(os.path.join(MODELS_DIR, 'Plant_Detection_Model.h5'))
except Exception as e:
    plant_disease_model = None
    print("Plant disease model not loaded:", e)

########################
#   Dynamic Data Loading for Fertilizer
########################
# Expected CSV path: dataset/fertilizer_data.csv
fertilizer_data_path = os.path.join(BASE_DIR, 'dataset', 'fertilizer_data.csv')
if os.path.exists(fertilizer_data_path):
    print("Fertilizer CSV found at:", fertilizer_data_path)
    fert_df = pd.read_csv(fertilizer_data_path)
    print("CSV Columns:", fert_df.columns)
    print("Unique values in 'Crop Type':", fert_df["Crop Type"].unique())
    unique_crop_types = sorted(list(fert_df["Crop Type"].unique()))
else:
    print("Fertilizer CSV NOT found. Using fallback list.")
    unique_crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy']

# Create numeric mapping for crop types for fertilizer prediction
crop_type_mapping = {crop: i for i, crop in enumerate(unique_crop_types)}

# Static mapping for soil types
soil_type_mapping = {
    'Sandy': 0,
    'Loamy': 1,
    'Black': 2,
    'Red': 3,
    'Clayey': 4
}

########################
#   Plant Disease Data
########################
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
         'description': "Apple scab is a fungal disease causing dark, scabby lesions on apple leaves and fruit.",
         'recommended_fertilizer': "Maintain balanced nutrition with adequate potassium to support plant vigor.",
         'recommended_treatment': "Apply a copper-based fungicide early in the season.",
         'usage_instructions': "Follow label directions; apply during early morning or late evening under wet conditions."
    },
    'Apple___Black_rot': {
         'description': "Black rot causes decay and dark lesions on apple fruits and leaves, leading to reduced quality.",
         'recommended_fertilizer': "Increase potassium levels and ensure balanced fertilization.",
         'recommended_treatment': "Use a bio-fungicide or chemical fungicide targeting black rot.",
         'usage_instructions': "Begin treatment at first signs of infection and follow manufacturer guidelines."
    },
    'Apple___Cedar_apple_rust': {
         'description': "Cedar apple rust affects apple trees, producing orange spots on leaves and fruit when in proximity to juniper.",
         'recommended_fertilizer': "Ensure overall balanced nutrition and trace elements.",
         'recommended_treatment': "Apply rust-specific fungicides once symptoms appear.",
         'usage_instructions': "Apply according to label recommendations during humid weather."
    },
    'Apple___healthy': {
         'description': "The apple tree is healthy and shows no visible disease symptoms.",
         'recommended_fertilizer': "Maintain a routine balanced fertilization program.",
         'recommended_treatment': "No treatment required.",
         'usage_instructions': "Follow standard cultural practices."
    },
    'Blueberry___healthy': {
         'description': "The blueberry plant is healthy and vigorous.",
         'recommended_fertilizer': "Use an acidic fertilizer like ammonium sulfate to maintain optimal pH.",
         'recommended_treatment': "No treatment needed.",
         'usage_instructions': "Ensure proper irrigation and pH management."
    },
    'Cherry___healthy': {
         'description': "The cherry tree is healthy with no signs of disease.",
         'recommended_fertilizer': "Apply a balanced fertilizer enriched with micronutrients.",
         'recommended_treatment': "No treatment necessary.",
         'usage_instructions': "Maintain routine care and monitoring."
    },
    'Cherry___Powdery_mildew': {
         'description': "Powdery mildew causes a white, powdery coating on cherry leaves and fruit.",
         'recommended_fertilizer': "Ensure balanced nutrition to keep the plant vigorous.",
         'recommended_treatment': "Apply sulfur-based fungicides.",
         'usage_instructions': "Spray early in the day and repeat as necessary during humid periods."
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
         'description': "This leaf spot disease appears as small gray or brown spots on corn leaves.",
         'recommended_fertilizer': "Balanced fertilization is important for disease resistance.",
         'recommended_treatment': "Apply a fungicide that targets leaf spots.",
         'usage_instructions': "Follow product instructions and apply during the early onset of symptoms."
    },
    'Corn___Common_rust': {
         'description': "Common rust manifests as small reddish-brown pustules on corn leaves.",
         'recommended_fertilizer': "Maintain adequate nitrogen and balanced nutrients.",
         'recommended_treatment': "Use fungicides containing strobilurins.",
         'usage_instructions': "Apply fungicide at first sign of rust; follow label instructions."
    },
    'Corn___Northern_Leaf_Blight': {
         'description': "Northern leaf blight produces elongated lesions on corn leaves.",
         'recommended_fertilizer': "Ensure balanced nutrient supply with emphasis on potassium.",
         'recommended_treatment': "Apply a fungicide effective against leaf blight.",
         'usage_instructions': "Spray at early symptom appearance; adhere to safety guidelines."
    },
    'Corn___healthy': {
         'description': "The corn plant is healthy with no visible symptoms.",
         'recommended_fertilizer': "Continue with routine fertilization as per soil test recommendations.",
         'recommended_treatment': "No treatment required.",
         'usage_instructions': "Follow standard agronomic practices."
    },
    'Grape___Black_rot': {
         'description': "Black rot in grapes causes dark, decaying spots on grape clusters and leaves.",
         'recommended_fertilizer': "Maintain balanced nutrition with an emphasis on potassium.",
         'recommended_treatment': "Apply a fungicide recommended for grape diseases.",
         'usage_instructions': "Apply fungicide as per label instructions at the onset of symptoms."
    },
    'Grape___Esca_(Black_Measles)': {
         'description': "Esca, or Black Measles, is a grapevine disease causing dark lesions and decline in vine vigor.",
         'recommended_fertilizer': "Use a balanced fertilizer with additional micronutrients.",
         'recommended_treatment': "Remove infected tissues and apply systemic fungicides if available.",
         'usage_instructions': "Consult local guidelines for precise treatment timing."
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
         'description': "Leaf blight in grapes appears as small dark spots that coalesce and cause leaf drop.",
         'recommended_fertilizer': "Maintain overall balanced nutrition.",
         'recommended_treatment': "Use a broad-spectrum fungicide.",
         'usage_instructions': "Apply during early infection stage and follow safety guidelines."
    },
    'Grape___healthy': {
         'description': "The grapevine is healthy with no disease symptoms.",
         'recommended_fertilizer': "Continue balanced nutrition with proper micronutrient supplementation.",
         'recommended_treatment': "No treatment necessary.",
         'usage_instructions': "Maintain regular care and monitoring."
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
         'description': "Citrus greening is a serious bacterial disease that causes yellowing of leaves, fruit drop, and tree decline.",
         'recommended_fertilizer': "Use citrus-specific fertilizers enriched with micronutrients.",
         'recommended_treatment': "No cure exists; focus on management practices and vector control.",
         'usage_instructions': "Implement integrated pest management and cultural practices to mitigate spread."
    },
    'Peach___Bacterial_spot': {
         'description': "Bacterial spot on peach appears as water-soaked lesions on leaves and fruit.",
         'recommended_fertilizer': "Ensure balanced nutrition with proper potassium levels.",
         'recommended_treatment': "Apply copper-based bactericides.",
         'usage_instructions': "Apply at first signs of infection; reapply as needed according to label."
    },
    'Peach___healthy': {
         'description': "The peach tree is healthy with no visible symptoms of disease.",
         'recommended_fertilizer': "Maintain a balanced fertilizer regimen.",
         'recommended_treatment': "No treatment required.",
         'usage_instructions': "Continue with standard cultural practices."
    },
    'Pepper,_bell___Bacterial_spot': {
         'description': "Bacterial spot in bell peppers causes dark, scabby lesions on leaves and fruits.",
         'recommended_fertilizer': "Maintain balanced nutrition.",
         'recommended_treatment': "Use copper-based bactericides.",
         'usage_instructions': "Apply at early disease onset; follow product instructions carefully."
    },
    'Pepper,_bell___healthy': {
         'description': "The bell pepper plant is healthy with no disease symptoms.",
         'recommended_fertilizer': "Use a balanced fertilizer formulated for peppers.",
         'recommended_treatment': "No treatment needed.",
         'usage_instructions': "Keep up with regular care and monitoring."
    },
    'Potato___Early_blight': {
         'description': "Early blight in potatoes causes brown, concentric lesions on leaves.",
         'recommended_fertilizer': "Ensure adequate potassium and overall balanced nutrition.",
         'recommended_treatment': "Apply fungicides effective against early blight.",
         'usage_instructions': "Spray as soon as symptoms appear; follow manufacturer's instructions."
    },
    'Potato___Late_blight': {
         'description': "Late blight is a severe disease causing rapid decay of potato foliage and tubers.",
         'recommended_fertilizer': "Maintain balanced nutrition to keep plants vigorous.",
         'recommended_treatment': "Apply a high-efficacy fungicide immediately.",
         'usage_instructions': "Follow label directions closely and remove infected parts to prevent spread."
    },
    'Potato___healthy': {
         'description': "The potato plant is healthy with no signs of disease.",
         'recommended_fertilizer': "Continue with fertilization based on soil test recommendations.",
         'recommended_treatment': "No treatment necessary.",
         'usage_instructions': "Maintain proper irrigation and pest control."
    },
    'Raspberry___healthy': {
         'description': "The raspberry plant is healthy, showing robust growth and fruiting.",
         'recommended_fertilizer': "Use organic fertilizers or balanced nutrient mixes.",
         'recommended_treatment': "No treatment needed.",
         'usage_instructions': "Ensure proper spacing and regular pruning."
    },
    'Soybean___healthy': {
         'description': "The soybean plant is healthy and producing well.",
         'recommended_fertilizer': "Maintain balanced nutrition, with focus on phosphorus and potassium.",
         'recommended_treatment': "No treatment required.",
         'usage_instructions': "Practice crop rotation and regular monitoring."
    },
    'Squash___Powdery_mildew': {
         'description': "Powdery mildew on squash produces a white, dusty coating on leaves.",
         'recommended_fertilizer': "Maintain overall balanced nutrition.",
         'recommended_treatment': "Apply sulfur-based fungicides.",
         'usage_instructions': "Apply early in the season and reapply during humid conditions."
    },
    'Strawberry___Leaf_scorch': {
         'description': "Leaf scorch in strawberries causes browning at the edges of leaves.",
         'recommended_fertilizer': "Use organic fertilizer to support soil health.",
         'recommended_treatment': "Apply organic remedies such as neem oil.",
         'usage_instructions': "Follow product guidelines and avoid over-watering."
    },
    'Strawberry___healthy': {
         'description': "The strawberry plant is healthy with vibrant leaves and productive fruiting.",
         'recommended_fertilizer': "Maintain balanced, preferably organic, fertilization.",
         'recommended_treatment': "No treatment necessary.",
         'usage_instructions': "Continue with regular care."
    },
    'Tomato___Bacterial_spot': {
         'description': "Bacterial spot in tomatoes causes dark lesions on leaves and fruit.",
         'recommended_fertilizer': "Ensure balanced nutrition with adequate potassium.",
         'recommended_treatment': "Apply copper-based bactericides.",
         'usage_instructions': "Start treatment at the first sign of spots and follow label instructions."
    },
    'Tomato___Early_blight': {
         'description': "Early blight in tomatoes is characterized by concentric, dark lesions on leaves.",
         'recommended_fertilizer': "Maintain balanced nutrition and supplement with potassium.",
         'recommended_treatment': "Apply fungicides effective against early blight.",
         'usage_instructions': "Spray early in the morning and repeat as per manufacturer recommendations."
    },
    'Tomato___Late_blight': {
         'description': "Late blight is a devastating disease that causes rapid decay of tomato foliage.",
         'recommended_fertilizer': "Ensure overall balanced nutrition for strong plant defense.",
         'recommended_treatment': "Use a high-efficacy fungicide immediately.",
         'usage_instructions': "Follow label directions carefully and remove infected parts promptly."
    },
    'Tomato___Leaf_Mold': {
         'description': "Leaf mold manifests as a moldy appearance on the underside of tomato leaves.",
         'recommended_fertilizer': "Maintain balanced fertilization.",
         'recommended_treatment': "Apply a fungicide specifically for leaf mold.",
         'usage_instructions': "Ensure adequate air circulation and follow product instructions."
    },
    'Tomato___Septoria_leaf_spot': {
         'description': "Septoria leaf spot causes small, dark spots on tomato leaves that may merge over time.",
         'recommended_fertilizer': "Keep nutrition balanced to reduce stress.",
         'recommended_treatment': "Apply a broad-spectrum fungicide.",
         'usage_instructions': "Follow application guidelines on the fungicide label."
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
         'description': "Spider mite infestations cause stippling and yellowing on tomato leaves.",
         'recommended_fertilizer': "Maintain plant health with balanced nutrition.",
         'recommended_treatment': "Use insecticidal soap or a miticide.",
         'usage_instructions': "Repeat treatments as needed and ensure thorough coverage."
    },
    'Tomato___Target_Spot': {
         'description': "Target spot produces concentric, target-like lesions on tomato leaves.",
         'recommended_fertilizer': "Maintain balanced nutrition, emphasizing potassium.",
         'recommended_treatment': "Apply a fungicide effective against target spot.",
         'usage_instructions': "Follow product instructions and apply at early symptom onset."
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
         'description': "This virus causes yellowing and curling of tomato leaves and can severely stunt growth.",
         'recommended_fertilizer': "Maintain balanced nutrition; stress reduction is crucial.",
         'recommended_treatment': "There is no chemical cure—focus on cultural practices.",
         'usage_instructions': "Remove infected plants and use resistant varieties if available."
    },
    'Tomato___Tomato_mosaic_virus': {
         'description': "Tomato mosaic virus causes a mosaic pattern on leaves and can distort plant growth.",
         'recommended_fertilizer': "Keep nutrition balanced to support plant health.",
         'recommended_treatment': "Remove severely infected plants and practice strict sanitation.",
         'usage_instructions': "Implement crop rotation and sanitize tools regularly."
    },
    'Tomato___healthy': {
         'description': "The tomato plant is healthy with no signs of disease.",
         'recommended_fertilizer': "Maintain balanced nutrition based on soil test recommendations.",
         'recommended_treatment': "No treatment necessary.",
         'usage_instructions': "Continue standard care and monitoring."
    }
}


########################
#   Weather API
########################



@app.route('/test-weather')
def test_weather():
    # Call your get_weather function directly
    data = get_weather(city="Hyderabad", country_code="IN")  # Or any other city/country code
    if data:
        return f"""
        <h2>Weather Test</h2>
        <p>Temperature: {data['temperature']} °C</p>
        <p>Humidity: {data['humidity']} %</p>
        <p>Rainfall: {data['rainfall']} mm</p>
        """
    else:
        return "<h2>Weather Test</h2><p>Failed to fetch weather data.</p>"



def get_weather(city="Hyderabad", country_code="91"):
    """
    Fetch current weather data from OpenWeatherMap.
    Returns a dict with temperature (°C), humidity (%), rainfall (mm).
    """
    API_KEY = os.getenv("API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather_data = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }
        return weather_data
    except Exception as e:
        print("Error fetching weather data:", e)
        return None


########################
#   Prediction Functions
########################
def predict_crop(input_data):
    """
    Predicts the best crop using input data.
    input_data: list -> [N, P, K, temperature, humidity, ph, rainfall]
    """
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = pd.DataFrame([input_data], columns=columns)
    scaled = crop_scaler.transform(df)
    pred = crop_model.predict(scaled)
    crop_name = crop_label_encoder.inverse_transform(pred)[0]
    return crop_name

def predict_crop_with_weather(user_data, city="Hyderabad", country_code="91"):
    """
    Integrates real-time weather data into the crop prediction.
    user_data: list -> [N, P, K, temperature, humidity, ph, rainfall]
    Overrides temperature, humidity, and (if zero) rainfall with real-time values.
    """
    weather = get_weather(city, country_code)
    if weather:
        user_data[3] = weather["temperature"]
        user_data[4] = weather["humidity"]
        if user_data[6] == 0:
            user_data[6] = weather["rainfall"]
    return predict_crop(user_data)


def predict_fertilizer_numeric(input_data):
    """
    Predicts fertilizer recommendation from numeric + categorical data.
    input_data keys:
      'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
      'Nitrogen', 'Potassium', 'Phosphorous'
    """
    df = pd.DataFrame([input_data])
    num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    df[num_cols] = fertilizer_scaler.transform(df[num_cols])
    pred = fertilizer_model.predict(df)
    if hasattr(fertilizer_label_encoder, 'inverse_transform'):
        fertilizer_name = fertilizer_label_encoder.inverse_transform(pred)[0]
    else:
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

def predict_plant_disease(image_path):
    """
    Predicts plant disease using the loaded Keras model.
    """
    if not plant_disease_model:
        return "No plant disease model loaded."
    try:
        image_obj = Image.open(image_path).convert('RGB')
        image_obj = image_obj.resize((224, 224))
        img_array = np.array(image_obj) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = plant_disease_model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        disease_name = plant_disease_class_names[class_idx]
        return disease_name
    except Exception as e:
        return f"Error in disease prediction: {str(e)}"

########################
#   Routes
########################
@app.route('/')
def index():
    return render_template('index.html')

# Crop Prediction Route with Weather Integration
@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    prediction = None
    weather_data = None
    if request.method == 'POST':
        try:
            # Gather user inputs
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            user_data = [N, P, K, temperature, humidity, ph, rainfall]
            
            # Check if the user wants to use real-time weather data
            use_weather = request.form.get('use_weather')
            if use_weather == "yes":
                weather_data = get_weather(city="Hyderabad", country_code="IN")
                if weather_data:
                    user_data[3] = weather_data["temperature"]
                    user_data[4] = weather_data["humidity"]
                    if user_data[6] == 0:
                        user_data[6] = weather_data["rainfall"]

            prediction = predict_crop(user_data)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('crop_prediction'))

    return render_template('crop_prediction.html', prediction=prediction, weather=weather_data)

# Fertilizer Prediction Route
@app.route('/fertilizer-prediction', methods=['GET', 'POST'])
def fertilizer_prediction():
    prediction = None
    weather_data = None
    if request.method == 'POST':
        try:
            soil_str = request.form['Soil Type']
            crop_str = request.form['Crop Type']
            temp = float(request.form['Temparature'])
            hum = float(request.form['Humidity'])
            moist = float(request.form['Moisture'])
            nitrogen = float(request.form['Nitrogen'])
            potassium = float(request.form['Potassium'])
            phosphorous = float(request.form['Phosphorous'])

            # Check if the user wants real-time weather
            use_weather = request.form.get('use_weather')
            if use_weather == "yes":
                weather_data = get_weather(city="Hyderabad", country_code="IN")
                if weather_data:
                    # Override temperature and humidity
                    temp = weather_data["temperature"]
                    hum = weather_data["humidity"]
                    # Note: There's no direct measure for "Moisture" from weather data,
                    # so you may decide how to handle it (leave as user input or set 0).
                    # For now, let's leave user moisture as is.

            # Build input data
            soil_numeric = soil_type_mapping.get(soil_str)
            crop_numeric = crop_type_mapping.get(crop_str)
            if soil_numeric is None or crop_numeric is None:
                flash("Invalid selection for Soil Type or Crop Type.")
                return redirect(url_for('fertilizer_prediction'))

            input_data = {
                'Temparature': temp,
                'Humidity': hum,
                'Moisture': moist,
                'Soil Type': soil_numeric,
                'Crop Type': crop_numeric,
                'Nitrogen': nitrogen,
                'Potassium': potassium,
                'Phosphorous': phosphorous
            }

            prediction = predict_fertilizer_numeric(input_data)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('fertilizer_prediction'))
    return render_template('fertilizer_prediction.html',
                           prediction=prediction,
                           weather=weather_data,
                           crop_types=unique_crop_types,
                           soil_types=list(soil_type_mapping.keys()))

# Plant Disease Detection Route (unchanged)
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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        disease_name = predict_plant_disease(file_path)
        disease_details = plant_disease_info.get(disease_name, {
            'description': "No information available.",
            'recommended_fertilizer': "No recommendation available.",
            'recommended_treatment': "No treatment available.",
            'usage_instructions': "No instructions available."
        })
        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('plant_disease.html',
                               disease_name=disease_name,
                               disease_desc=disease_details['description'],
                               recommended_fertilizer=disease_details['recommended_fertilizer'],
                               recommended_treatment=disease_details.get('recommended_treatment'),
                               usage_instructions=disease_details.get('usage_instructions'),
                               image_url=image_url)
    return render_template('plant_disease.html')

########################
#   Main
########################
if __name__ == '__main__':
    app.run(debug=True)
