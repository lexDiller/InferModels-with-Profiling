import os
import time
import psutil
import GPUtil
from statistics import mean
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import torch.profiler as profiler


class YOLODataset(Dataset):
    def __init__(self, data_directory, transform=None):
        super().__init__()
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


def get_size_in_gb(bytes_size):
    return bytes_size / (1024 * 1024 * 1024)


def yolo_inference(model_path, dataset, batch_size=16, device_name="cuda", model_name=None, img_size=None):
    device = torch.device(device_name)
    model = YOLO(model_path)
    model.to(device)

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Creating and starting the profiler
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if device_name == "cuda" else None,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True
    )
    prof.start()

    model.eval()
    with torch.no_grad():
        for images in dataset_loader:
            results = model(images, device=device, verbose=False)

    prof.stop()

    # Save profiling results
    filename = f"profiling_runs/{model_name}_{batch_size}_{img_size}_{device_name}.txt"
    with open(filename, 'w') as f:
        f.write("Detailed Layer Profiling Results:\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if device_name == "cuda" else "cpu_time_total",
            row_limit=500
        ))

    return filename


if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])

    dataset = YOLODataset(
        "/home/moo/PycharmProjects/Yolo_inference/warm_1000",
        transform=test_transform
    )

    profile_file = yolo_inference(
        model_path="66_train.pt",
        dataset=dataset,
        batch_size=32,
        device_name="cuda",
        model_name='yolov8n-cls',
        img_size='128'
    )

    print(f"\nYOLO model profiling results saved to: {profile_file}")

