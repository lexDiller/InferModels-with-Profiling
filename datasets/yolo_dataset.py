import os
from PIL import Image
from torch.utils.data import Dataset

class YOLODataset(Dataset):
    def __init__(self, data_directory, transform=None):
        self.data_directory = data_directory
        self.transform = transform
        self.images = [os.path.join(data_directory, img_name)
                       for img_name in os.listdir(data_directory)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
