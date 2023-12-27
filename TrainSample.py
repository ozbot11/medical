from multiprocessing import freeze_support
import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
    
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

trainSet = torchvision.datasets.CIFAR100("./cifar", train = True, transform=preprocess, download=True)
testSet = torchvision.datasets.CIFAR100("./cifar", train = False, transform=preprocess, download=True)

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
        model.load_state_dict(weights.to(device))

    for epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            y_pred = model(inputs.to(device))
            y_label = torch.zeros_like(y_pred, device=device)
            # print(f'len: {len(y_pred)}')
            for j in range(len(y_pred)):
                y_label[j, labels[j]] = 1

            # print(f'i: {i}, inputs_shape: {inputs.shape}, labels_shape {labels.shape}, y_pref.shape: {y_pred.shape}, y_labels_shape {y_label.shape}, ')
            loss = loss_fn(y_pred, y_label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, step {i}/{train_dataloader.__len__()}')
            if i > 1000:
                break
        
        print(f'Finished epoch {epoch}, latest loss {loss}')

        with torch.no_grad():
            loss = 0.0
            for i, (inputs, labels) in enumerate(test_dataloader):
                y_pred = model(inputs.to(device))
                y_label = torch.zeros_like(y_pred)
                for j in range(len(y_pred)):
                    y_label[j, labels[j]] = 1

                loss += loss_fn(y_pred, y_label)
                if i % 100 == 0:
                    print(f'Epoch {epoch}, test step {i}/{test_dataloader.__len__()}')
                if i > 1000:
                    break
            average_loss = loss / test_dataloader.__len__()

            print(f'average_loss {average_loss}')

        print(f'Saving to {SAVE_PATH}')
        model.train()
        torch.save(model.state_dict(), SAVE_PATH)

if __name__ == '__main__':
    main()