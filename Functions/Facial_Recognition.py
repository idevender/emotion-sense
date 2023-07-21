__author__ = "JasonLuo"
import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def model_loading(model_path = ""):
    model = models.densenet169(pretrained=True).to(device)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 3)
    )
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

def image_transform(img):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).to(device)

def facial_recognize(image_path, CNN_model):
    img = image_transform(Image.open(image_path).convert('RGB'))
    outcome = CNN_model(img.unsqueeze(dim=0))

    return outcome


def face(image_path):
    model = model_loading("76Test.pth")
    outputs = facial_recognize(image_path=image_path,CNN_model=model)
    _, preds = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)

    label_map = {0: 'Angry', 1: 'Happy', 2: 'Sad'}
    result = label_map.get(preds.item())

    return result

