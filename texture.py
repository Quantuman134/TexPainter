import torch
import torch.nn as nn

class Texture(nn.Module):
    def __init__(self, size=(512, 512), is_latent=False, device='cpu') -> None:
    # sample mode: 'bilinear', 'nearest'
        super().__init__()
        self.device = device
        self.size = size
        self.height = size[0]
        self.width = size[1]
        self.is_latent = is_latent

        self.texture = None
        self.set_gaussian()

    def set_gaussian(self, mean=0, sig=1):
        if self.is_latent:
            self.texture = torch.randn((self.height, self.width, 4), dtype=torch.float32, device=self.device, requires_grad=True) * sig + mean
        else:
            self.texture = torch.randn((self.height, self.width, 3), dtype=torch.float32, device=self.device, requires_grad=True) * sig + mean

    def set_indentical_value(self, value=0):
        if self.is_latent:
            self.texture = torch.ones((self.height, self.width, 4), dtype=torch.float32, device=self.device, requires_grad=True) * value
        else:
            self.texture = torch.ones((self.height, self.width, 3), dtype=torch.float32, device=self.device, requires_grad=True) * value
