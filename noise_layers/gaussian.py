import torch.fft
import torch.nn as nn
from torchvision.transforms import GaussianBlur


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

class GaussianLPF(nn.Module):
    """
    Applies a gaussian (in the frequency domain) to the input signal.
    """
    def __init__(self, sigma):
        super(GaussianLPF, self).__init__()
        self.sigma = sigma

    def forward(self, noise_and_cover):
        noised_image = noise_and_cover[0]
        noised_transform = torch.fft.fftshift(torch.fft.fft2(noised_image))
        m = torch.fft.fftshift(torch.fft.fftfreq(noised_transform.size(dim=2)))
        n = torch.fft.fftshift(torch.fft.fftfreq(noised_transform.size(dim=3)))
        l, k = torch.meshgrid(n, m)
        d = torch.sqrt(k ** 2 + l ** 2)
        gaussian_lpf = torch.exp(-0.5 * (d / self.sigma) ** 2)
        noised_transform = gaussian_lpf * noised_transform
        noise_and_cover[0] = torch.real(torch.fft.ifft2(torch.fft.ifftshift(noised_transform)))
        return noise_and_cover
