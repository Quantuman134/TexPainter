import argparse
import torch
import utils
from utils import device
from stable_diffusion_depth import StableDiffusionDepth
from texture import Texture
from function import tex_paint, tex_export

def main(args):
    seed = args.seed
    utils.seed_everything(seed)

    mesh_dir = args.mesh_dir
    save_dir = args.save_dir
    
    text_prompt = args.text_prompt

    # diffusion configuration
    num_inference_steps = args.num_inference_steps
    end_step = args.end_step
    guidance_scale = args.guidance_scale

    sd = StableDiffusionDepth(device=device, num_inference_steps=num_inference_steps)
    print("[INFO] Stable Diffusion Loaded")

    text_embeddings = sd.text_encoding(text_prompt)
    DM_params = DMParams(text_embeddings, guidance_scale, end_step)

    # optimization
    epochs = args.opt_eps
    learning_rate = args.opt_lr

    # rendering configuration
    # mesh
    mesh_data = utils.mesh_import(mesh_dir=mesh_dir, device=device)

    if mesh_data is None:
        print("[INFO] Mesh data load failed")
    print("[INFO] Mesh Loaded")

    # texture
    latent_tex_size = args.latent_tex_size
    rgb_tex_size = args.rgb_tex_size
    tex_latent = Texture(size=(latent_tex_size, latent_tex_size), device=device, is_latent=True)
    tex_rgb = Texture(size=(rgb_tex_size, rgb_tex_size), device=device, is_latent=False)

    # cameras
    camera_num = args.camera_num
    fov = args.fov
    dist = args.dist
    dists = torch.ones((camera_num, 1), dtype=torch.float32, device=device) * dist
    elevs = torch.tensor([args.elev] * camera_num , dtype=torch.float32, device=device).reshape(camera_num, 1)
    azims = torch.tensor([i*(360.0/camera_num) for i in range(camera_num)], dtype=torch.float32, device=device).reshape(camera_num, 1)

    camera_params = CameraParams(camera_num, dists, elevs, azims, fov)

    # texture painting
    tex_paint(mesh_data, tex_latent, tex_rgb, DM_params, sd, camera_params, epochs, learning_rate)

    tex_export(mesh_data, tex_rgb, camera_params, device=device, save_dir=save_dir)

class CameraParams():
    def __init__(self, camera_num, dists, elevs, azims, fov) -> None:
        self.camera_num = camera_num
        self.dists = dists
        self.elevs = elevs
        self.azims = azims
        self.fov = fov

class DMParams():
    def __init__(self, text_embeddings, guidance_scale, end_step) -> None:
        self.text_embeddings = text_embeddings
        self.guidance_scale = guidance_scale
        self.end_step = end_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--text_prompt', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--end_step', type=int, default=35)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--latent_tex_size', type=int, default=2000)
    parser.add_argument('--rgb_tex_size', type=int, default=600)
    parser.add_argument('--opt_eps', type=int, default=20)
    parser.add_argument('--opt_lr', type=float, default=0.1)

    parser.add_argument('--camera_num', type=int, default=8)
    parser.add_argument('--fov', type=int, default=35)
    parser.add_argument('--dist', type=float, default=1.5)
    parser.add_argument('--elev', type=float, default=30)
    
    args = parser.parse_args()

    main(args)