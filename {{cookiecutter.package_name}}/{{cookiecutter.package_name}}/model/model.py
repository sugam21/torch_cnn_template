from torchvision.models import mobilenet_v3_large
import torch.nn as nn
from {{cookiecutter.package_name}}.base import BaseModel
from {{cookiecutter.package_name}}.utils import get_logger

LOG = get_logger("model")


class MobileNet(BaseModel):
    def __init__(self, out_feature: int):
        super().__init__()
        self.model = mobilenet_v3_large(weights="DEFAULT", progress=True)
        self.model.classifier[3] = nn.Linear(1280, out_feature, bias=True)
        LOG.debug("Successfully loaded the model.")

    def forward(self, image):
        model_output = self.model(image)
        return nn.Softmax(dim=1)(model_output)
