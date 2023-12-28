from multiprocessing import freeze_support
import os
import os.path
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from torchvision.io import read_image

SAVE_PATH = "weights.pth"
batch_size = 10
n_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = {
    "NORMAL": 0,
    "DRUSEN": 1,
    "CNV": 2,
    "DME": 3
}

class CustomImageDataset(Dataset):
    def __init__(self, folder_name, transform, train):
        self.folder_name = folder_name
        self.transform = transform
        self.train = train
        self.files = []
        self.labels = []

        if train == True:
            self.folder_name = os.path.join(self.folder_name, "train", "train")
        else:
            self.folder_name = os.path.join(self.folder_name, "validation", "validation")

        for label_folder in os.listdir(self.folder_name):
            for file_name in os.listdir(os.path.join(self.folder_name, label_folder)):
                full_file_name = os.path.join(self.folder_name, label_folder, file_name)
                self.files.append(full_file_name)
                self.labels.append(classes[label_folder])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label


trainSet = CustomImageDataset("./archive", transform=preprocess, train=True)
testSet = CustomImageDataset("./archive", transform=preprocess, train=False)
# trainSet = torchvision.datasets.CIFAR100("./cifar", train = True, transform=preprocess, download=True)
# testSet = torchvision.datasets.CIFAR100("./cifar", train = False, transform=preprocess, download=True)

train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=2)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=2)

def main():
    model = torchvision.models.densenet201(weights=None)
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    if os.path.exists(SAVE_PATH):
        weights = torch.load(SAVE_PATH)
        model.load_state_dict(weights)

    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, train_dataloader, loss_fn, optimizer, device)
        find_loss(model, test_dataloader, loss_fn, device)
        print(f'Saving to {SAVE_PATH}')
        torch.save(model.state_dict(), SAVE_PATH)

def train_one_epoch(epoch: int, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        y_label = torch.zeros_like(outputs, device=device)
        for j in range(len(outputs)):
            y_label[j, labels[j]] = 1

        loss = loss_fn(outputs, y_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f'Step {i}/{len(dataloader)}, Loss: {loss.item()}')

    average_loss = total_loss / len(dataloader)
    
    print(f'Finished epoch {epoch}, latest loss {average_loss}')

    print(f'average_loss {average_loss}')
    
    return average_loss

def find_loss(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    with torch.no_grad():
        loss = 0.0
        for i, (inputs, labels) in enumerate(test_dataloader):
            y_pred = model(inputs.to(device))
            y_label = torch.zeros_like(y_pred)
            for j in range(len(y_pred)):
                y_label[j, labels[j]] = 1

            loss += loss_fn(y_pred, y_label)
            if i % 100 == 0:
                print(f'Test step {i}/{test_dataloader.__len__()}')
        average_loss = loss / test_dataloader.__len__()

        return average_loss

if __name__ == '__main__':
    main()

