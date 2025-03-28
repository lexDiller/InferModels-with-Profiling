import torch
from PIL import Image
from torchvision import transforms
import torch.profiler as profiler
from ultralytics import YOLO

def cold_start_inference_yolo(image_path, model_path, device_name="cuda", model_name=None, img_size=None):
    device = torch.device(device_name)
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

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model = YOLO(model_path)
    model.to(device)
    with torch.no_grad():
        results = model(image, device=device)
    prof.stop()

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
        image_path="path/to/your/image.jpg",
        model_path="path/to/your/66_train.pt",
        device_name="cuda",
        model_name='yolov8n',
        img_size='128'
    )
    print(f"Profiling results saved to: {profile_file}")
