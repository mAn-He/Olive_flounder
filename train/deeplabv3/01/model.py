"""
DeepLabV3 Model Definition.

This script provides a function to create a DeepLabV3 model
with a ResNet101 backbone, pre-trained on a subset of COCO and Pascal VOC.
The classifier head is customized for a specific number of output channels.
"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def create_deep_lab_v3(output_channels=3):
    """
    Creates a DeepLabV3 model with a ResNet101 backbone and a custom classifier head.

    The model is pre-trained on a subset of COCO train2017, specifically on the
    20 categories that are present in the Pascal VOC dataset. The classifier head
    is replaced with a new DeepLabHead configured for the specified number of
    output channels.

    Args:
        output_channels (int, optional): The number of output channels
            (classes) for the segmentation task. Defaults to 3.

    Returns:
        torch.nn.Module: The configured DeepLabV3 model, set to training mode.
    """
    # Load pre-trained DeepLabV3 model with ResNet101 backbone
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=False)
    # Replace the classifier head for the custom number of output channels
    # The head of the DeepLabV3 model is the final layer that produces the segmentation map.
    # Its input channels must match the output channels of the backbone (2048 for ResNet101).
    model.classifier = DeepLabHead(2048, output_channels)
    
    # Set the model to training mode by default
    # This is important because some layers (like batch normalization) behave differently
    # during training versus evaluation.
    model.train()
    return model
