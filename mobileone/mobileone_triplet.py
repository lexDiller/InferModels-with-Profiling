import time
import torch
import torch.nn as nn
from .mobileone import mobileone

class MobileOneWithTriplet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        # Инициализируем базовую модель (вариант s4)
        self.base = mobileone(variant='s4', num_classes=1000)
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            self.base.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint,
                                      strict=False)
            print("Loaded pretrained weights from:", pretrained_path)
        # Получаем размерность признаков и создаём новый классификатор
        in_features = self.base.linear.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.base.linear = nn.Identity()

    def forward(self, x):
        start_time = time.time()
        x = self.base.stage0(x)
        start_time = time.time()
        x = self.base.stage1(x)
        start_time = time.time()
        x = self.base.stage2(x)
        start_time = time.time()
        x = self.base.stage3(x)
        start_time = time.time()
        features = self.base.stage4(x)
        features = self.base.gap(features)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits
