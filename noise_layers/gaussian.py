from torchvision.transforms import GaussianBlur
import torch.nn as nn

class GaussianFilter(nn.Module):
    """
    Applies a gaussian filter to the input signal.
    """
    def __init__(self, kernel_size, sigma):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)(noised_image)
        return noised_and_cover
