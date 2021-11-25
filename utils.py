import json
import io
import glob
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)