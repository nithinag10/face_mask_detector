from torchvision import models
import torch
import torch.nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 3)
PATH ='trained_model_freezing.pth'
model.load_state_dict(torch.load(PATH))
model.eval()

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


def transform_image(image):
    transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    outputs = model(image_tensor)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()







