import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Learning rate 
LEARNING_RATE = 0.002
BATCH_SIZE = 32
NUM_EPOCHS = 50
MODEL_PATH = 'backend/model/emotion_model.pth'

# Define the Dataset class to handle the images
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        for label, folder_name in enumerate(['Angry', 'Happy', 'Sad']):
            folder_path = os.path.join(self.root_dir, folder_name)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"Found image: {file_path}, Label: {
                          label}")  # Debugging print
                    self.image_list.append(file_path)
                    self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_list[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = self.label_list[idx]
            return img, label
        except Exception as e:
            print(f'Error loading image {self.image_list[idx]}: {e}')
            return None, None


# Define image transformation and normalization
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model definition


def get_densenet_model(num_classes=3):
    model = models.densenet169(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

# Image transformation function for inference
def image_transform(img):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(img).to(device)

# Function to load the model for inference


def load_model(model_path):
    model = get_densenet_model(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def facial_recognize(image_path, model):
    img = image_transform(Image.open(image_path).convert('RGB'))
    outputs = model(img.unsqueeze(dim=0))  # Add batch dimension
    _, preds = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)

    label_map = {0: 'Angry', 1: 'Happy', 2: 'Sad'}
    result = label_map.get(preds.item())
    return result


def train_model(root_dir, model_save_path=MODEL_PATH, num_epochs=NUM_EPOCHS):
    dataset = EmotionDataset(root_dir, transform=transform)
    print(f"Total number of images: {len(dataset)}")
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_densenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Epoch [{
              epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_dataset)
        print(f'Validation Accuracy: {val_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


def predict_emotion(image_path):
    model = load_model(MODEL_PATH)
    return facial_recognize(image_path, model)


if __name__ == "__main__":
    # Training the model (example usage)
    train_model('../data')

    # Testing the model with an image (example usage)
    emotion = predict_emotion('../data/Sad/0x0.jpg')
    print(f'Predicted emotion: {emotion}')
