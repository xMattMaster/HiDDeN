from torchvision.transforms.functional import rotate
import torch.nn as nn


class Rotate(nn.Module):
    """
    Rotates the input signal.
    """
    def __init__(self, angle):
        super(Rotate, self).__init__()
        self.angle = angle

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = rotate(noised_image, self.angle)
        noised_and_cover[0] = noised_image
        return noised_and_cover
