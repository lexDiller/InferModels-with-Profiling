import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.profiler as profiler
from ultralytics import YOLO
from datasets.yolo_dataset import YOLODataset

def yolo_inference(model_path, dataset, batch_size=16, device_name="cuda", model_name=None, img_size=None):
    device = torch.device(device_name)
    model = YOLO(model_path)
    model.to(device)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    prof = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA if device_name == "cuda" else None,
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
            _ = model(images, device=device, verbose=False)
    prof.stop()
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
    dataset = YOLODataset("path/to/your/dataset_directory", transform=test_transform)
    profile_file = yolo_inference(
        model_path="path/to/your/66_train.pt",
        dataset=dataset,
        batch_size=32,
        device_name="cuda",
        model_name='yolov8n-cls',
        img_size='128'
    )
    print(f"Profiling results saved to: {profile_file}")
