
import torch

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

def load_model():
    model = torch.nn.Identity()  # placeholder
    return model

def predict(model, image):
    return "NV", 0.85
