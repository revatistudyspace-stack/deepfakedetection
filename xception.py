import torch
import torch.nn as nn
import torchvision.models as models


def xception_model():
    # Use MobileNet v2 as a lightweight proxy model for the demo.
    # Set pretrained=False for offline/CI environments; change to True if weights are available.
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model
