import torch
import torch.nn as nn
from torchvision import models

def load_model(num_classes=29, model_path="best_model.pth", device="cpu"):
    """
    Load EfficientNet-B3 model for classification.

    Args:
        num_classes (int): Number of output classes.
        model_path (str): Path to saved model weights.
        device (str): 'cpu' or 'cuda'

    Returns:
        model: PyTorch model ready for evaluation.
    """
    # Load EfficientNet-B3 with pretrained weights
    model = models.efficientnet_b3(pretrained=True)
    in_features = model.classifier[1].in_features

    # Replace classifier for our specific number of classes
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1536),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1536, num_classes)
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
