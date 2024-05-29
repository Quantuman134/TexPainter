from typing import Optional, Union
from pytorch3d.renderer.mesh import shader
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import interpolate_face_attributes
import torch

class Shader_Output():
    def __init__(self, images, pixel_uvs=None) -> None:
        self.images = images
        self.pixel_uvs = pixel_uvs

class DiffShader(shader.ShaderBase):
    def __init__(self,
        diff_tex = None,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
        mesh_data = None,
        shading_method = 'phong', 
        background_map = None
        ):
        super().__init__(device, cameras, lights, materials, blend_params)
        self.diff_tex = diff_tex
        self.mesh = mesh_data['mesh_obj']
        self.verts = mesh_data['verts']
        self.aux = mesh_data['aux']
        self.faces = mesh_data['faces']
        self.shading_method = shading_method
        self.background_map = background_map
        self.device = device

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> Shader_Output:
        cameras = super()._get_cameras(**kwargs)
        texels, pixel_uvs = self.texture_sample(fragments=fragments)

        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        if self.shading_method == 'phong':
            colors = shader.phong_shading(
                meshes=meshes,
                fragments=fragments,
                texels=texels[:, :, :, :, 0:3],
                lights=lights,
                cameras=cameras,
                materials=materials
            )
        elif self.shading_method == 'depth':
            colors = self.depth_shading(fragments=fragments)
        else:
            colors = texels

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = self.softmax_rgb_blend_custom(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )

        shader_output = Shader_Output(images=images, pixel_uvs=pixel_uvs)

        return shader_output

    def uv_cal(self, fragments: Fragments):
        packing_list = [
            i[j] for i, j in zip([self.aux.verts_uvs.to(self.device)], [self.faces.textures_idx.to(self.device)])
        ]
        faces_verts_uvs = torch.cat(packing_list)

        pixel_uvs = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs).reshape(-1, 2) #range [0, 1]

        pixel_uvs = pixel_uvs * 2.0 - 1.0 #range [-1, 1]
        temps = pixel_uvs[:, 0].clone()
        pixel_uvs[:, 0] = -pixel_uvs[:, 1]
        pixel_uvs[:, 1] = temps

        return pixel_uvs

    def texture_sample(self, fragments: Fragments):
        N, H, W, K = fragments.pix_to_face.size()

        pixel_uvs = self.uv_cal(fragments=fragments)

        colors = self.diff_tex(pixel_uvs)
        if colors.size()[1] == 3:
            colors = (colors + 1) / 2
        elif colors.size()[1] == 8:
            colors[:, 0:5] = (colors[:, 0:5] + 1) / 2
        colors = colors.reshape(N, H, W, colors.size()[1])
        texels = colors.unsqueeze(3).repeat(1, 1, 1, K, 1)
        return (texels, pixel_uvs)
    
    def depth_shading(self, fragments):
        return self.z_coordinate_color(fragments=fragments)
    

    def softmax_rgb_blend_custom(
        self,
        colors: torch.Tensor,
        fragments,
        blend_params: BlendParams,
        znear: Union[float, torch.Tensor] = 1.0,
        zfar: Union[float, torch.Tensor] = 100
    ) -> torch.Tensor:
        """
        RGB and alpha channel blending to return an RGBA image based on the method
        proposed in [1]
        - **RGB** - blend the colors based on the 2D distance based probability map and
            relative z distances.
        - **A** - blend based on the 2D distance based probability map.

        Args:
            colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
            fragments: namedtuple with outputs of rasterization. We use properties
                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - dists: FloatTensor of shape (N, H, W, K) specifying
                the 2D euclidean distance from the center of each pixel
                to each of the top K overlapping faces.
                - zbuf: FloatTensor of shape (N, H, W, K) specifying
                the interpolated depth from each pixel to to each of the
                top K overlapping faces.
            blend_params: instance of BlendParams dataclass containing properties
                - sigma: float, parameter which controls the width of the sigmoid
                function used to calculate the 2D distance based probability.
                Sigma controls the sharpness of the edges of the shape.
                - gamma: float, parameter which controls the scaling of the
                exponential function used to control the opacity of the color.
                - background_color: (3) element list/tuple/torch.Tensor specifying
                the RGB values for the background color.
            znear: float, near clipping plane in the z direction
            zfar: float, far clipping plane in the z direction

        Returns:
            RGBA pixel_colors: (N, H, W, 4)

        [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
        Image-based 3D Reasoning'
        """

        N, H, W, K = fragments.pix_to_face.shape
        D = (colors.size()[4] + 1) #pixel buffer depth
        pixel_colors = torch.ones((N, H, W, D), dtype=colors.dtype, device=colors.device)
        #pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
        background_color = torch.ones(D-1, dtype=colors.dtype, device=colors.device) * 1.0

        if self.background_map is not None:
            background_color =  self.background_map.permute(0, 2, 3, 1)

        #background_color = _get_background_color(blend_params, fragments.pix_to_face.device)

        # Weight for background color
        eps = 1e-10

        # Mask for padded pixels.
        mask = fragments.pix_to_face >= 0

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        alpha = torch.prod((1.0 - prob_map), dim=-1)

        # Weights for each face. Adjust the exponential by the max z to prevent
        # overflow. zbuf shape (N, H, W, K), find max over K.
        # TODO: there may still be some instability in the exponent calculation.

        # Reshape to be compatible with (N, H, W, K) values in fragments
        if torch.is_tensor(zfar):
            # pyre-fixme[16]
            zfar = zfar[:, None, None, None]
        if torch.is_tensor(znear):
            # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
            #  `__getitem__`.
            znear = znear[:, None, None, None]

        z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1)[..., None] + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
        weighted_background = delta * background_color
        pixel_colors[..., :(D-1)] = (weighted_colors + weighted_background) / denom
        pixel_colors[..., D-1] = 1.0 - alpha

        return pixel_colors


    
