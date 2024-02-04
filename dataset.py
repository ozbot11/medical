import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
            self.folder_name = os.path.join(self.folder_name, "train")
        else:
            self.folder_name = os.path.join(self.folder_name, "test")

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