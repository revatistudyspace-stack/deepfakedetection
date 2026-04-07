import torch
import numpy as np
import cv2
from xception import xception_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = xception_model()
model.to(device)
model.eval()

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))  # MobileNet uses 224
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return torch.tensor(frame, dtype=torch.float32).to(device)

def predict(frame):
    tensor = preprocess(frame)
    
    with torch.no_grad():
        output = model(tensor)
    
    prob = torch.softmax(output, dim=1)
    return float(prob[0][1])