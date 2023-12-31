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
from time import time
import argparse
from dataset import CustomImageDataset

SAVE_PATH = "densenet201.pth"
batch_size = 10
n_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainSet = CustomImageDataset("./archive", transform=preprocess, train=True)
testSet = CustomImageDataset("./archive", transform=preprocess, train=False)

train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory_device="cuda:0", pin_memory=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory_device="cuda:0", pin_memory=True)

def classify(model: nn.Module, file_name: str, device: torch.device):
    im = Image.open(file_name).convert("RGB")
    im = preprocess(im).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(im)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    class_names = list(classes.keys())
    predicted_class_name = class_names[predicted_class.item()]

    print(f'predicted_class_name {predicted_class_name} confidence.item {confidence.item()}')

    return predicted_class_name, confidence.item()

def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, device: torch.device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    prev_test_loss = find_loss(model, test_dataloader, loss_fn, device)
    print(f'Starting test loss: {prev_test_loss}')
    for epoch in range(n_epochs):
        start = time()
        training_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        end = time()
        test_loss = find_loss(model, test_dataloader, loss_fn, device)
        train_accurracy = find_accuracy(model, train_dataloader, device)
        test_accurracy = find_accuracy(model, test_dataloader, device)
        print(f'Epoch {epoch}, train acurracy: {train_accurracy*100}%, test accuracy: {test_accurracy*100}%, training loss: {training_loss}, test loss: {test_loss}, time: {end-start}')
        if test_loss < prev_test_loss:
            print(f'Saving to {SAVE_PATH}')
            torch.save(model.state_dict(), SAVE_PATH)
            prev_test_loss = test_loss

def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        y_label = torch.zeros_like(outputs)
        for j in range(len(outputs)):
            y_label[j, labels[j]] = 1

        loss = loss_fn(outputs, y_label.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0 and i != 0:
            print(f'  Step {i}/{len(dataloader)}, Loss: {loss.item()}')

    average_loss = total_loss / len(dataloader)   
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
        average_loss = loss / test_dataloader.__len__()

        return average_loss

def find_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.to(device)

            # Perform inference
            outputs = model(inputs).cpu()
            for i in range(len(outputs)):
                o = outputs[i]
                l = labels[i]
                total_samples += 1
                if (torch.argmax(o).item() == l.item()):
                    correct_predictions += 1

        accuracy = correct_predictions / total_samples
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Medical Image Classification')
    parser.add_argument('mode', choices=['train', 'classify'], help='Mode: train or classify')
    parser.add_argument('--file', help='Path to the image file (required for classify mode)')

    args = parser.parse_args()

    if args.mode == 'train':
        model = torchvision.models.densenet201(weights=None)

        if os.path.exists(SAVE_PATH):
            print(f'loading weights from {SAVE_PATH}')
            weights = torch.load(SAVE_PATH)
            model.load_state_dict(weights)

        model.to(device)
        train(model, train_dataloader, test_dataloader, device)

    elif args.mode == 'classify':
        if args.file is None:
            print("Error: --file is required for classify mode.")
        else:
            model = torchvision.models.densenet201(weights=None)
            if os.path.exists(SAVE_PATH):
                print(f'loading weights from {SAVE_PATH}')
                weights = torch.load(SAVE_PATH)
                model.load_state_dict(weights)

            model.to(device)
            classify(model, args.file, device)

if __name__ == '__main__':
    main()