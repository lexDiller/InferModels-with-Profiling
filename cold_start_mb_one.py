import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms
import time
import psutil
import GPUtil
from mobileone import reparameterize_model, mobileone


class MobileOneWithTriplet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        # Инициализируем базовую модель с 1000 классами (как в ImageNet)
        self.base = mobileone(variant='s4', num_classes=1000)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            # Некоторые ключи в state_dict могут отличаться, поэтому используем strict=False
            self.base.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint,
                                      strict=False)
            print("Loaded pretrained weights from:", pretrained_path)

        # Получаем размерность признаков из последнего слоя
        in_features = self.base.linear.in_features

        # Создаем новый классификатор для нашего количества классов
        self.classifier = nn.Linear(in_features, num_classes)

        # Заменяем оригинальный классификатор на Identity
        self.base.linear = nn.Identity()

    def forward(self, x):
        # features = self.base(x)
        start_time = time.time()
        x = self.base.stage0(x)
        stage0_time = time.time() - start_time

        start_time = time.time()
        x = self.base.stage1(x)
        stage1_time = time.time() - start_time

        start_time = time.time()
        x = self.base.stage2(x)
        stage2_time = time.time() - start_time

        start_time = time.time()
        x = self.base.stage3(x)
        stage3_time = time.time() - start_time

        start_time = time.time()
        features = self.base.stage4(x)
        stage4_time = time.time() - start_time

        features = self.base.gap(features)
        features = torch.flatten(features, 1)

        start_time = time.time()
        logits = self.classifier(features)
        classifier_time = time.time() - start_time

        # print(f"Stage 0 time: {stage0_time:.6f} seconds")
        # print(f"Stage 1 time: {stage1_time:.6f} seconds")
        # print(f"Stage 2 time: {stage2_time:.6f} seconds")
        # print(f"Stage 3 time: {stage3_time:.6f} seconds")
        # print(f"Stage 4 time: {stage4_time:.6f} seconds")
        # print(f"Classifier time: {classifier_time:.6f} seconds")

        return features, logits


def get_size_in_gb(bytes_size):
    return bytes_size / (1024 * 1024 * 1024)


def cold_start_inference(image_path, model_path, device_name="cuda", use_reparameterization=False, model_name=None,
                         img_size=None):
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
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Model initialization and loading
    model = MobileOneWithTriplet(num_classes=105).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    if use_reparameterization:
        model = reparameterize_model(model)
        model = model.to(device)

    model.eval()

    # Inference
    with torch.no_grad():
        _, logits = model(image)
        _, preds = torch.max(logits, 1)

    prof.stop()

    # Save profiling results
    filename = f"profiling_runs/cold_{model_name}_{img_size}_{device_name}_{'reparam' if use_reparameterization else 'standard'}.txt"
    with open(filename, 'w') as f:
        f.write("Detailed Layer Profiling Results:\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if device_name == "cuda" else "cpu_time_total",
            row_limit=500
        ))

    return filename, preds.cpu().item()


if __name__ == "__main__":
    # Run inference with standard model
    standard_profile_file, standard_pred = cold_start_inference(
        image_path="/home/moo/PycharmProjects/Yolo_inference/cold_3/image__2.jpg",
        model_path="transfered.pt",
        device_name="cuda",
        use_reparameterization=False,
        model_name='Mb1_s4',
        img_size='128'
    )

    print(f"\nStandard model profiling results saved to: {standard_profile_file}")
    print(f"Standard model prediction: {standard_pred}")
    print("\n" + "=" * 50 + "\n")

    # Run inference with reparameterized model
    reparam_profile_file, reparam_pred = cold_start_inference(
        image_path="/home/moo/PycharmProjects/Yolo_inference/cold_3/image__2.jpg",
        model_path="transfered.pt",
        device_name="cuda",
        use_reparameterization=True,
        model_name='Mb1_s4',
        img_size='128'
    )

    print(f"\nReparameterized model profiling results saved to: {reparam_profile_file}")
    print(f"Reparameterized model prediction: {reparam_pred}")

