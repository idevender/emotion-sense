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
     r"/*": {"origins": ["http://localhost:3001", "http://127.0.0.1:3001"]}})

# Define the model


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Load the trained model
model = EmotionCNN()
# After loading the model
sample_input = torch.randn(1, 1, 48, 48)  # Create a random tensor
with torch.no_grad():
    sample_output = model(sample_input)
print("Sample prediction:", sample_output)

try:
    model.load_state_dict(torch.load(
        'model/emotion_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
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
        img = Image.open(io.BytesIO(img_bytes))
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
