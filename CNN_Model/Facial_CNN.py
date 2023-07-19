import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd
import warnings

# Set the learning rate range for the learning rate finder
LEARNING_RATE = 0.002
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        for label, folder_name in enumerate(['Angry','Happy','Sad']):
            folder_path = os.path.join(self.root_dir, folder_name)
            for root, directories, files in os.walk(folder_path):
                for file in files:
                    self.image_list.append(os.path.join(root, file))
                    self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert('RGB')


        if self.transform:
            img = self.transform(img)

        label = self.label_list[idx]
        return img, label


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0

def main():
    dataset = Dataset(r'C:\Users\Woody\Desktop\facial_recognition', transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet169(pretrained=True).to(device)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 3)
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # Split the data into train and test sets

    train_indices, test_indices = train_test_split(list(range(int(len(dataset)))), test_size=0.2, random_state=123)


    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False,num_workers=4)


    num_epochs = 80


    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        try:
            for img,label in tqdm(train_dataloader):
                model.train()
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)

                loss = criterion(outputs, label)

                _, preds = torch.max(outputs, 1)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                running_loss += loss.item() * (img.size(0))
                running_corrects += torch.sum(preds == label.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / (len(train_dataset))

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            model.eval()  # Set the model to evaluation mode
            # #
            running_corrects = 0

            # # # evaluate the result at each epoch

            with torch.no_grad():
                for img , label in tqdm(test_dataloader):
                    img = img.to(device)
                    label = label.to(device)

                    outputs = model(img)

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == label.data)

            epoch_acc = running_corrects.double() / (len(test_dataset))
            print(f'Acc: {epoch_acc:.4f}')


        except Exception as e:
            print(e)



if __name__ == '__main__':
    main()