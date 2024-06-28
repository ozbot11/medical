import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import densenet201
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset import CustomImageDataset
import torch.nn as nn
import torch.optim as optim
from time import time
import argparse
import os

SAVE_PATH = "densenet201.pth"
batch_size = 10
n_epochs = 100

classes = {
    "NORMAL": 0,
    "DRUSEN": 1,
    "CNV": 2,
    "DME": 3
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainSet = CustomImageDataset("./archive", transform=preprocess, train=True)
testSet = CustomImageDataset("./archive", transform=preprocess, train=False)

# Does Not Work: trainSet = CustomImageDataset("./kermany/OCT2017", transform=preprocess, train=True)
# Does Not Work: testSet = CustomImageDataset("./kermany/OCT2017", transform=preprocess, train=False)

train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def grad_cam(model, image, target_layer):
    model.eval()
    image = image.unsqueeze(0).to(device)

    def forward_hook(module, input, output):
        model._features_output = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)
    output = model(image)
    handle.remove()

    model.zero_grad()
    target_class = output.argmax(dim=1).item()
    target = output[0][target_class]
    target.backward()

    gradients = model._features_output.grad
    activations = model._features_output

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.size(2), image.size(3)))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image = np.uint8(255 * image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img

# def classify(model: nn.Module, file_name: str, device: torch.device):
#     im = Image.open(file_name).convert("RGB")
#     im = preprocess(im).to(device)
    
#     model.eval()

#     with torch.no_grad():
#         output = model(im.unsqueeze(0))
#         probabilities = torch.nn.functional.softmax(output, dim=1)
#         confidence, predicted_class = torch.max(probabilities, 1)
    
#     class_names = list(classes.keys())
#     predicted_class_name = class_names[predicted_class.item()]

#     target_layer = model.features.denseblock4
#     overlay = grad_cam(model, im, target_layer)

#     cv2.imshow("Grad-CAM", overlay)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print(f'predicted_class_name: {predicted_class_name}, confidence: {confidence.item()}')

#     return predicted_class_name, confidence.item()

def classify(model: nn.Module, file_name: str, device: torch.device):
    im = Image.open(file_name).convert("RGB")
    im = preprocess(im).to(device)
    
    model.eval()

    with torch.no_grad():
        output = model(im.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    class_names = list(classes.keys())
    predicted_class_name = class_names[predicted_class.item()]

    # Print the diagnosis before displaying the Grad-CAM
    print(f'Predicted Class: {predicted_class_name}')
    print(f'Confidence: {confidence.item()}')

    target_layer = model.features.denseblock4
    overlay = grad_cam(model, im, target_layer)

    # Display the Grad-CAM overlay
    cv2.imshow("Grad-CAM", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        train_accuracy = find_accuracy(model, train_dataloader, device)
        test_accuracy = find_accuracy(model, test_dataloader, device)
        print(f'Epoch {epoch}, train accuracy: {train_accuracy*100}%, test accuracy: {test_accuracy*100}%, training loss: {training_loss}, test loss: {test_loss}, time: {end-start}')
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
        for i, (inputs, labels) in enumerate(dataloader):
            y_pred = model(inputs.to(device))
            y_label = torch.zeros_like(y_pred)
            for j in range(len(y_pred)):
                y_label[j, labels[j]] = 1

            loss += loss_fn(y_pred, y_label)
        average_loss = loss / len(dataloader)

        return average_loss

def find_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.to(device)

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
        model = densenet201(weights=None)

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
            model = densenet201(weights=None)
            if os.path.exists(SAVE_PATH):
                print(f'loading weights from {SAVE_PATH}')
                weights = torch.load(SAVE_PATH)
                model.load_state_dict(weights)

            model.to(device)
            classify(model, args.file, device)

if __name__ == '__main__':
    main()


# import cv2
# import os
# import os.path
# import argparse
# import numpy as np

# from PIL import Image
# from dataset import CustomImageDataset, classes
# from time import time
# from multiprocessing import freeze_support
# from typing import Tuple

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torchvision.models import densenet201
# from torchvision import transforms
# from torchvision.io import read_image

# SAVE_PATH = "densenet201.pth"
# batch_size = 10
# n_epochs = 100

# classes = {
#     "NORMAL": 0,
#     "DRUSEN": 1,
#     "CNV": 2,
#     "DME": 3
# }

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# trainSet = CustomImageDataset("./kermany/OCT2017", transform=preprocess, train=True)
# testSet = CustomImageDataset("./kermany/OCT2017", transform=preprocess, train=False)

# train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# def grad_cam(model, image, target_layer):
#     model.eval()
#     image = image.unsqueeze(0).to(device)

#     def forward_hook(module, input, output):
#         model._features_output = output
#         output.retain_grad()

#     handle = target_layer.register_forward_hook(forward_hook)
#     output = model(image)
#     handle.remove()

#     model.zero_grad()
#     target_class = output.argmax(dim=1).item()
#     target = output[0][target_class]
#     target.backward()

#     gradients = model._features_output.grad
#     activations = model._features_output

#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
#     for i in range(activations.size(1)):
#         activations[:, i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(activations, dim=1).squeeze()
#     heatmap = F.relu(heatmap)
#     heatmap /= torch.max(heatmap)

#     heatmap = heatmap.detach().cpu().numpy()
#     heatmap = cv2.resize(heatmap, (image.size(2), image.size(3)))

#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
#     image = np.uint8(255 * image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

#     return superimposed_img

# def classify(model: nn.Module, file_name: str, device: torch.device):
#     im = Image.open(file_name).convert("RGB")
#     im = preprocess(im).to(device)
    
#     model.eval()

#     with torch.no_grad():
#         output = model(im.unsqueeze(0))
#         probabilities = torch.nn.functional.softmax(output, dim=1)
#         confidence, predicted_class = torch.max(probabilities, 1)
    
#     class_names = list(classes.keys())
#     predicted_class_name = class_names[predicted_class.item()]

#     # Print the diagnosis before displaying the Grad-CAM
#     print(f'predicted_class_name {predicted_class_name} confidence.item {confidence.item()}')


#     target_layer = model.features.denseblock4
#     overlay = grad_cam(model, im, target_layer)

#     # Display the Grad-CAM overlay
#     cv2.imshow("Grad-CAM", overlay)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return predicted_class_name, confidence.item()


# def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, device: torch.device):
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)

#     prev_test_loss = find_loss(model, test_dataloader, loss_fn, device)
#     print(f'Starting test loss: {prev_test_loss}')
#     for epoch in range(n_epochs):
#         start = time()
#         training_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
#         end = time()
#         test_loss = find_loss(model, test_dataloader, loss_fn, device)
#         train_accuracy = find_accuracy(model, train_dataloader, device)
#         test_accuracy = find_accuracy(model, test_dataloader, device)
#         print(f'Epoch {epoch}, train accuracy: {train_accuracy*100}%, test accuracy: {test_accuracy*100}%, training loss: {training_loss}, test loss: {test_loss}, time: {end-start}')
#         if test_loss < prev_test_loss:
#             print(f'Saving to {SAVE_PATH}')
#             torch.save(model.state_dict(), SAVE_PATH)
#             prev_test_loss = test_loss

# def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
#     model.train()
#     total_loss = 0.0

#     for i, (inputs, labels) in enumerate(dataloader):
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad(set_to_none=True)

#         outputs = model(inputs)
#         y_label = torch.zeros_like(outputs)
#         for j in range(len(outputs)):
#             y_label[j, labels[j]] = 1

#         loss = loss_fn(outputs, y_label.to(device))
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         if i % 100 == 0 and i != 0:
#             print(f'  Step {i}/{len(dataloader)}, Loss: {loss.item()}')

#     average_loss = total_loss / len(dataloader)   
#     return average_loss

# def find_loss(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
#     with torch.no_grad():
#         loss = 0.0
#         for i, (inputs, labels) in enumerate(dataloader):
#             y_pred = model(inputs.to(device))
#             y_label = torch.zeros_like(y_pred)
#             for j in range(len(y_pred)):
#                 y_label[j, labels[j]] = 1

#             loss += loss_fn(y_pred, y_label)
#         average_loss = loss / len(dataloader)

#         return average_loss

# def find_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device):
#     model.eval()
#     correct_predictions = 0
#     total_samples = 0

#     with torch.no_grad():
#         for _, (inputs, labels) in enumerate(dataloader, 0):
#             inputs = inputs.to(device)

#             outputs = model(inputs).cpu()
#             for i in range(len(outputs)):
#                 o = outputs[i]
#                 l = labels[i]
#                 total_samples += 1
#                 if (torch.argmax(o).item() == l.item()):
#                     correct_predictions += 1

#         accuracy = correct_predictions / total_samples
#     return accuracy

# def main():
#     parser = argparse.ArgumentParser(description='Medical Image Classification')
#     parser.add_argument('mode', choices=['train', 'classify'], help='Mode: train or classify')
#     parser.add_argument('--file', help='Path to the image file (required for classify mode)')

#     args = parser.parse_args()

#     if args.mode == 'train':
#         model = densenet201(weights=None)

#         if os.path.exists(SAVE_PATH):
#             print(f'loading weights from {SAVE_PATH}')
#             weights = torch.load(SAVE_PATH)
#             model.load_state_dict(weights)

#         model.to(device)
#         train(model, train_dataloader, test_dataloader, device)

#     elif args.mode == 'classify':
#         if args.file is None:
#             print("Error: --file is required for classify mode.")
#         else:
#             model = densenet201(weights=None)
#             if os.path.exists(SAVE_PATH):
#                 print(f'loading weights from {SAVE_PATH}')
#                 weights = torch.load(SAVE_PATH)
#                 model.load_state_dict(weights)

#             model.to(device)
#             classify(model, args.file, device)

# if __name__ == '__main__':
#     main()