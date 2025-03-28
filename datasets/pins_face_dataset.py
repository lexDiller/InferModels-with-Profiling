import os
from PIL import Image
from torch.utils.data import Dataset

class PinsFaceDataset(Dataset):
    def __init__(self, data_directory, is_inference=False, transform=None):
        self.data_directory = data_directory
        self.transform = transform
        self.is_inference = is_inference

        if not self.is_inference:
            self.classes = os.listdir(data_directory)
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.images = []
            self.labels = []
            for class_name in self.classes:
                class_dir = os.path.join(data_directory, class_name)
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        else:
            self.images = [os.path.join(data_directory, img_name)
                           for img_name in os.listdir(data_directory)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if not self.is_inference:
            label = self.labels[idx]
            return image, label
        return image
