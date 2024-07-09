from flask import Flask, request, jsonify, render_template, send_file, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

Labels = ['Benign', 'Malignant', 'Non MRI']
Descriptions = {
    'Benign': 'No tumor detected.', 
    'Malignant': 'A malignant tumor is detected. Immediate medical attention is recommended.', 
    'Non MRI': 'Please add the brain MRI images if possible.'
}

# Load the trained model with custom objects
model_path = "api/model/model.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Define function to preprocess image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Normalize pixel values
    return img

# Define function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["jpg", "jpeg", "png"]

# Route for uploading image and making predictions
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        # Check if file is empty
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        # Check if file is an image
        if file and allowed_file(file.filename):
            # Save the uploaded image to the "uploads" directory
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            
            # Read the image file
            img = cv2.imread(file_path)
            # Preprocess the image
            img = preprocess_image(img)
            # Make prediction
            prediction = model.predict(np.expand_dims(img, axis=0))
            # Format prediction
            class_idx = np.argmax(prediction)
            class_label = Labels[class_idx]
            confidence = float(prediction[0][class_idx])

            # The image path for the template should be the route that serves the image
            image_url = url_for('serve_image', filename=file.filename)

            # Get description for the predicted class
            description = Descriptions.get(class_label, 'No description available.')

            # Return result page with prediction, confidence, description, and image path
            return render_template("result.html", 
                                   prediction={"class": class_label, "confidence": confidence, "description": description}, 
                                   image_path=image_url)

        else:
            return jsonify({"error": "Invalid file type"})

    return render_template("index.html")

# Route to serve uploaded image
@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_file(os.path.join("uploads", filename))

if __name__ == "__main__":
    app.run()
