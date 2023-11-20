from typing import Optional

from torchvision.models import resnet18
from torch.nn import Linear
from torchvision.models._api import Weights


def get_model(name: str, weights: Optional[Weights]):
    if name == 'resnet18':
        model = resnet18(weights=weights)
        model.fc = Linear(in_features=512, out_features=149, bias=True)
    else:
        raise ValueError(f'Unsupported model "{name}"')

    return model
