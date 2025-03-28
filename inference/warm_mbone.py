import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.profiler as profiler
from mobileone.mobileone_triplet import MobileOneWithTriplet
from mobileone import reparameterize_model
from datasets.pins_face_dataset import PinsFaceDataset

def inference(checkpoint_path, num_classes, dataset, batch_size=8, device_name="cuda", use_reparameterization=False, model_name=None, img_size=None):
    device = torch.device(device_name)
    model = MobileOneWithTriplet(num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    if use_reparameterization:
        model = reparameterize_model(model)
        model = model.to(device)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
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
    with torch.no_grad():
        for images, _ in dataset_loader:
            images = images.to(device)
            _, logits = model(images)
            torch.max(logits, 1)
    prof.stop()
    filename = f"profiling_runs/{model_name}_{batch_size}_{img_size}_{device_name}_{'reparam' if use_reparameterization else 'standard'}.txt"
    with open(filename, 'w') as f:
        f.write("Detailed Layer Profiling Results:\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if device_name == "cuda" else "cpu_time_total",
            row_limit=500
        ))
    return filename

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = PinsFaceDataset("path/to/your/dataset_directory", is_inference=True, transform=test_transform)
    profile_file = inference(
        checkpoint_path="path/to/your/transfered.pt",
        num_classes=105,
        dataset=dataset,
        batch_size=16,
        device_name="cuda",
        use_reparameterization=False,
        model_name='Mb1_s4',
        img_size='128'
    )
    print(f"Profiling results saved to: {profile_file}")
