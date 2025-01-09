import torch
from PIL import Image
from torchvision import transforms
import time
import psutil
import GPUtil
from ultralytics import YOLO


def get_size_in_gb(bytes_size):
    return bytes_size / (1024 * 1024 * 1024)


def cold_start_inference_yolo(image_path, model_path, device_name="cuda", model_name=None, img_size=None):
    device = torch.device(device_name)

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

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Model initialization and loading
    model = YOLO(model_path)
    model.to(device)

    # Inference
    with torch.no_grad():
        results = model(image, device=device)

    prof.stop()

    # Save profiling results
    filename = f"profiling_runs/cold_{model_name}_{img_size}_{device_name}.txt"
    with open(filename, 'w') as f:
        f.write("Detailed Layer Profiling Results:\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if device_name == "cuda" else "cpu_time_total",
            row_limit=500
        ))

    return filename, results[0]


if __name__ == "__main__":
    profile_file, results = cold_start_inference_yolo(
        image_path="/home/moo/PycharmProjects/Yolo_inference/cold_3/image__2.jpg",
        model_path="66_train.pt",
        device_name="cuda",
        model_name='yolov8n',
        img_size='128'
    )

    print(f"\nYOLO cold start profiling results saved to: {profile_file}")
