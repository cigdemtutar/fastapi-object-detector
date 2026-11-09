import json

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ObjectDetector(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.baseModel = base_model
        self.numClasses = num_classes

        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        features = self.baseModel(x)
        return self.regressor(features), self.classifier(features)


class Detector:
    def __init__(self, weights_path, classes_path):
        with open(classes_path) as f:
            self.classes = json.load(f)

        self.model = ObjectDetector(resnet50(weights=None), len(self.classes))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def predict(self, image_bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Gecerli bir resim dosyasi degil")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))
        tensor = self.transform(torch.from_numpy(image)).unsqueeze(0)

        with torch.no_grad():
            boxes, logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            prob, idx = probs.max(dim=-1)

        sx, sy, ex, ey = boxes[0].tolist()
        return {
            "label": self.classes[idx.item()],
            "probability": round(prob.item(), 4),
            "bbox": {"start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey},
        }
