import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = tf.keras.models.load_model('optimized_model.h5')

# Define a function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image before making predictions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array

# Function to map the predicted index to class labels
def predict_image(img_array):
    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]

    # Load class indices (Make sure you have a class_indices.json file or list available)
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Reverse the class_indices dictionary to get the class label from the predicted index
    class_labels = {v: k for k, v in class_indices.items()}

    # Return the predicted class label
    return class_labels.get(predicted_class_idx, "Unknown")

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            # Secure the filename and save the file to the uploads directory
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            # Preprocess the image and predict the class
            img_array = preprocess_image(img_path)
            predicted_class = predict_image(img_array)

            # Return the prediction result as JSON
            return jsonify({"predicted_class": predicted_class})
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
