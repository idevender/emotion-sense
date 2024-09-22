# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import sys
import traceback

app = Flask(__name__)
CORS(app, resources={
     r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# Define the model

# Model path
MODEL_PATH = '../model_training/emotion_model.pth'

# Load the DenseNet model


def get_densenet_model(num_classes=3):
    model = torch.hub.load('pytorch/vision', 'densenet169', pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


# Initialize model
model = get_densenet_model()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define emotions
emotions = ['Angry', 'Happy', 'Sad']


@app.route('/predict', methods=['POST'])
def predict():
    print("Received a prediction request")
    if 'image' not in request.files:
        print("No image found in the request")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        print("No image selected")
        return jsonify({'error': 'No image selected'}), 400

    try:
        print("Image received:", file.filename)
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        print("Image opened successfully.")
        img = transform(img)
        print("Image transformed successfully.")
        img = img.unsqueeze(0)  # Add batch dimension
        print("Image reshaped for model input.")

        with torch.no_grad():
            outputs = model(img)
            print("Model inference completed.")
            _, predicted = torch.max(outputs.data, 1)
            emotion = emotions[predicted.item()]
            print(f'Predicted emotion: {emotion}')
        return jsonify({'emotion': emotion})
    except UnidentifiedImageError:
        print("Error: The uploaded file is not a valid image.")
        return jsonify({'error': 'Invalid image file.'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Error processing the image.'}), 500


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend is working'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
