from diff_renderer import DiffRenderer
from utils import device
from texture import Texture
from tqdm.auto import tqdm
import torch

def tex_paint(
        tex_latent,
        tex_rgb,
        DM_params,
        diffusion_model,
        camera_params
    ):
    text_embeddings = DM_params.text_embeddings
    guidance_scale = DM_params.guidance_scale
    end_step = DM_params.end_step

    camera_num = camera_params.camera_num
    dists = camera_params.dists
    elevs = camera_params.elevs
    azims = camera_params.azims
    fov = camera_params.fov

    renderer = DiffRenderer(device=device)

    tex_Q_latent = Texture(size=tex_latent.size, device=device, is_latent=True) 
    tex_Q_rgb = Texture(size=tex_rgb.size, device=device, is_latent=False) 

    for i in tqdm(range(0, end_step)):
        # fetch time step in sampler
        time_step = diffusion_model.scheduler.timesteps[i]
        tex_Q_latent.set_indentical_value(0.0)
        tex_Q_rgb.set_indentical_value(0.0)

        img_latent_list = []
        img_Q_latent_list = []
        pixel_uvs_latent_list = []
        img_Q_rgb_list = []
        pixel_uvs_rgb_list = []

        for n in range(camera_num):
            dist = dists[n]
            elev = elevs[n]
            azim = azims[n]

            # depth map
            depth_map = depth_map_rendering()


def depth_map_rendering(renderer:DiffRenderer, mesh_data, tex_latent, elev, azim, dist, fov):
    '''
    depth map: [1, 3, 64, 64], (-1, 1)
    '''
    renderer.rasterization_setting(image_size=64)
    renderer.camera_setting(dist=dist, elev=elev, azim=azim, fov=fov)
    background_map = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device=tex_latent.device)

    image = renderer.rendering(mesh_data=mesh_data, diff_tex=tex_latent, 
                                shading_method='depth', extra_output=False, 
                                background_map=background_map)
    depth_map = (image[:, :, :, 0] * 2 - 1.0).unsqueeze(-1).permute(0, 3, 1, 2)
    return depth_map