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
PATH ='trained_model.pth'
model.load_state_dict(torch.load(PATH))
model.eval()

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


def transform_image(image_bytes):
    transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    outputs = model(image_tensor)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    classes = {
        "0": "Please wear Mask! pay your fine",
        "1": "Thank You for wearning mask keep distance from others",
        "2": "Sariyag Hakko Guruve"
    }
    return classes[str(predicted.item())]


# with open("frame70.jpg", "rb") as image:
#   f = image.read()
#   img =transform_image(f)
#   outputs = model(img)
#   # max returns (value ,index)
#   _, predicted = torch.max(outputs.data, 1)
#   classes = {
#       "0": "Please wear Mask! pay your fine",
#       "1": "Thank You for wearning mask keep distance from others",
#       "2": "Sariyag Hakko Guruve"
#   }
#   print(classes[str(predicted.item())])






