"""
DeepLabV3 Model Definition for '012' type datasets.

This script provides a function to create a DeepLabV3 model
with a ResNet101 backbone. Unlike the '01' version, this one is configured
typically without pre-trained weights for the backbone by default,
and the classifier head is customized for a specific number of output channels
(usually 3 for '012' type data).
"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def create_deep_lab_v3(output_channels=3): # Renamed from createDeepLabv3 for consistency
    """
    Creates a DeepLabV3 model with a ResNet101 backbone and a custom classifier head.

    The model is initialized **without** pre-trained weights for the ResNet101 backbone
    by default in this configuration (pretrained=False). The classifier head
    is replaced with a new DeepLabHead configured for the specified number of
    output channels.

    Args:
        output_channels (int, optional): The number of output channels
            (classes) for the segmentation task. Defaults to 3, suitable for
            '012' type datasets (e.g., background, class1, class2).

    Returns:
        torch.nn.Module: The configured DeepLabV3 model, set to training mode.
    """
    # Load DeepLabV3 model with ResNet101 backbone.
    # Note: `pretrained=False` means the backbone is not initialized with COCO weights.
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, 
                                                    progress=False)
    
    # Replace the classifier head for the custom number of output channels.
    # The head of the DeepLabV3 model is the final layer that produces the segmentation map.
    # Its input channels must match the output channels of the backbone (2048 for ResNet101).
    model.classifier = DeepLabHead(2048, output_channels)
    
    # Set the model to training mode by default.
    # This is important because some layers (like batch normalization) behave differently
    # during training versus evaluation.
    model.train()
    return model
