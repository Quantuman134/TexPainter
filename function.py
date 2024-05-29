from diff_renderer import DiffRenderer
from utils import device
from texture import Texture
from tqdm.auto import tqdm

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

