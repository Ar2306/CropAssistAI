from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved scaler and label encoder
with open("models/mlmodels/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/mlmodels/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("models/mlmodels/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/crop-recommendation", methods=["POST"])
def crop_recommendation():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        
        # Scale the input data
        scaled_features = scaler.transform(features)
        
        # Predict the crop
        prediction = model.predict(scaled_features)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({"recommended_crop": crop_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)