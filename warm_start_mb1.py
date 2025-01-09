import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from mobileone import reparameterize_model, mobileone
import time
import torch.profiler as profiler
from statistics import mean


class PinsFaceDataset(Dataset):
    def __init__(self, data_directory, is_inference=False, transform=None):
        super().__init__()
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


class MobileOneWithTriplet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        self.base = mobileone(variant='s4', num_classes=1000)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            self.base.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint,
                                      strict=False)
            print("Loaded pretrained weights from:", pretrained_path)

        # Получаем размерность признаков из последнего слоя
        in_features = self.base.linear.in_features
        self.classifier = nn.Linear(in_features, num_classes)
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


def inference(checkpoint_path, num_classes, dataset, batch_size=8, device_name="cuda", use_reparameterization=False, model_name=None, img_size=None):
    device = torch.device(device_name)
    model = MobileOneWithTriplet(num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))

    if use_reparameterization:
        model = reparameterize_model(model)
        model = model.to(device)

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()

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

    with torch.no_grad():
        for image in dataset_loader:
            image = image.to(device)
            _, logits = model(image)
            _, preds = torch.max(logits, 1)

    prof.stop()

    # Save detailed profiling results to a file
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

    dataset = PinsFaceDataset("/home/moo/PycharmProjects/Yolo_inference/warm_1000",
                              is_inference=True,
                              transform=test_transform)

    # Run inference with standard model
    standard_profile_file = inference(
        checkpoint_path="transfered.pt",
        num_classes=105,
        dataset=dataset,
        batch_size=16,
        device_name="cuda",
        use_reparameterization=False,
        model_name='Mb1_s4',
        img_size='128'
    )

    print(f"\nStandard model profiling results saved to: {standard_profile_file}")
    print("\n" + "=" * 50 + "\n")

    # Run inference with reparameterized model
    reparam_profile_file = inference(
        checkpoint_path="transfered.pt",
        num_classes=105,
        dataset=dataset,
        batch_size=16,
        device_name="cuda",
        use_reparameterization=True,
        model_name='Mb1_s4',
        img_size='128'
    )

    print(f"\nReparameterized model profiling results saved to: {reparam_profile_file}")
