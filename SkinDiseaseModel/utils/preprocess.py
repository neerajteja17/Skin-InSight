
import numpy as np
import cv2
import torch

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)
