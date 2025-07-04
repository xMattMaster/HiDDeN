import torch.fft
import torch.nn as nn


class IdealLowPassFilter(nn.Module):
    """
    Applies an ideal low-pass filter to the input signal.
    """
    def __init__(self, cutoff_frequency):
        super(IdealLowPassFilter, self).__init__()
        self.cutoff = cutoff_frequency

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_transform = torch.fft.fftshift(torch.fft.fft2(noised_image))
        m = torch.fft.fftshift(torch.fft.fftfreq(noised_transform.size(dim=2)))
        n = torch.fft.fftshift(torch.fft.fftfreq(noised_transform.size(dim=3)))
        l, k = torch.meshgrid(n, m)
        d = torch.sqrt(k ** 2 + l ** 2)
        mask = torch.abs(d) <= self.cutoff
        noised_transform = mask * noised_transform
        noised_and_cover[0] = torch.real(torch.fft.ifft2(torch.fft.ifftshift(noised_transform)))
        return noised_and_cover
