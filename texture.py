import torch
import torch.nn as nn
import numpy as np
import tinycudann as tcnn

class Texture(nn.Module):
    def __init__(self, size=(512, 512), is_latent=False, device='cpu') -> None:
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

    def set_image(self, img_tensor):
        #img_tensor size: [B, C, H, W], color value [0, 1]
        B, C, H, W = img_tensor.size()
        img_tensor = img_tensor * 2 - 1.0
        self.height = H
        self.width = W
        img_tensor = img_tensor.to(self.device)
        self.texture = img_tensor.squeeze().permute(1, 2, 0)

    def forward(self, uvs):
        colors = self.texture_sample(uvs)
        return colors
    
    def texture_sample(self, uvs):
        uvs = (uvs + 1)/2
        uvs[:, 0] *= (self.width - 1)
        uvs[:, 1] *= (self.height - 1)

        us_0 = uvs[:, 0].floor().type(torch.int32)
        us_1 = uvs[:, 0].ceil().type(torch.int32)
        vs_0 = uvs[:, 1].floor().type(torch.int32)
        vs_1 = uvs[:, 1].ceil().type(torch.int32)

        a = (uvs[:, 0] - us_0).reshape(-1, 1)
        b = (uvs[:, 1] - vs_0).reshape(-1, 1)

        a[a<0.5] = 0
        a[a>=0.5] = 1
        b[b<0.5] = 0
        b[b>=0.5] = 1

        us_0[us_0 >= self.width] = self.width - 1
        us_1[us_1 >= self.width] = self.width - 1
        vs_0[vs_0 >= self.height] = self.height - 1
        vs_1[vs_1 >= self.height] = self.height - 1

        colors = self.texture[us_0, vs_0, :] * (1-a) * (1-b) + self.texture[us_1, vs_0, :] * a * (1-b) \
            + self.texture[us_0, vs_1, :] * (1-a) * b + self.texture[us_1, vs_1, :] * a * b

        return colors
    
    def texel_set(self, uvs, colors):
        '''
        uvs: [N, 2]
        colors: [N, 3]
        '''
        # only neareast is feasible currently
        uvs = (uvs + 1)/2
        uvs[:, 0] *= (self.width - 1)
        uvs[:, 1] *= (self.height - 1)

        us_0 = uvs[:, 0].floor().type(torch.int32)
        us_1 = uvs[:, 0].ceil().type(torch.int32)
        vs_0 = uvs[:, 1].floor().type(torch.int32)
        vs_1 = uvs[:, 1].ceil().type(torch.int32)

        a = (uvs[:, 0] - us_0).reshape(-1, 1)
        b = (uvs[:, 1] - vs_0).reshape(-1, 1)

        a[a<0.5] = 0
        a[a>=0.5] = 1
        b[b<0.5] = 0
        b[b>=0.5] = 1
        a = a.type(torch.int32)
        b = b.type(torch.int32)

        us = (us_0.reshape(-1, 1) * (1-a) + us_1.reshape(-1, 1) * a).squeeze()
        vs = (vs_0.reshape(-1, 1) * (1-b) + vs_1.reshape(-1, 1) * b).squeeze()

        self.texture[us, vs, :] = colors
            
    def texel_fetch(self, uvs):
        '''
        uvs: [N, 2]
        colors: [N, 3]
        '''
        # only neareast is feasible currently
        uvs = (uvs + 1)/2
        uvs[:, 0] *= (self.width - 1)
        uvs[:, 1] *= (self.height - 1)

        us_0 = uvs[:, 0].floor().type(torch.int32)
        us_1 = uvs[:, 0].ceil().type(torch.int32)
        vs_0 = uvs[:, 1].floor().type(torch.int32)
        vs_1 = uvs[:, 1].ceil().type(torch.int32)

        a = (uvs[:, 0] - us_0).reshape(-1, 1)
        b = (uvs[:, 1] - vs_0).reshape(-1, 1)

        a[a<0.5] = 0
        a[a>=0.5] = 1
        b[b<0.5] = 0
        b[b>=0.5] = 1
        a = a.type(torch.int32)
        b = b.type(torch.int32)

        us = (us_0.reshape(-1, 1) * (1-a) + us_1.reshape(-1, 1) * a).squeeze()
        vs = (vs_0.reshape(-1, 1) * (1-b) + vs_1.reshape(-1, 1) * b).squeeze()

        colors = self.texture[us, vs, :]

        return colors
        

class NeuralTextureField(nn.Module):
    def __init__(self, width=512, depth=3, input_dim=2, expected_tex_size = 512, 
                 pe_enable=True, sampling_disturb=False, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.width = width
        self.depth = depth
        self.pe_enable = pe_enable
        self.sampling_disturb = sampling_disturb
        self.input_dim = input_dim
        self.output_dim = 3
        self.expectex_tex_size = expected_tex_size
        layers = []
        
        if pe_enable:
            desired_resolution = 4096
            num_levels = 16
            base_grid_resolution = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale                
            }
            pe = tcnn.Encoding(input_dim, encoding_config, dtype=torch.float32)
            layers.append(pe)
            #layers.append(nn.Linear(pe.mapping_size, width, bias=False))
            layers.append(nn.Linear(pe.n_output_dims, width, bias=False))
        else:
            layers.append(nn.Linear(input_dim, width, bias=False))  

        for i in range(depth - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(width, width, bias=False))

        layers.append(nn.ReLU())  
        layers.append(nn.Linear(width, self.output_dim, bias=False))
        self.base = nn.ModuleList(layers)

        self.to(device)
        print("[NEURAL TEXTURE INFO:]")
        print(self.base)

    def forward(self, x):
        if self.sampling_disturb:
            x += (torch.rand_like(x) - 0.5) * (2.0/self.expectex_tex_size)

        for layer in self.base:
            x = layer(x)
        colors = x
        colors = torch.clamp(colors, -0.95, 0.95)
        return colors