from diff_renderer import DiffRenderer
from utils import device
from texture import Texture, NeuralTextureField
from tqdm.auto import tqdm
from stable_diffusion_depth import StableDiffusionDepth
import torch
import torch.nn as nn
import time
import utils

def tex_paint(
        mesh_data,
        tex_latent,
        tex_rgb,
        DM_params,
        diffusion_model:StableDiffusionDepth,
        camera_params,
        epochs,
        learning_rate
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
            depth_map = depth_map_rendering(renderer, mesh_data, tex_latent, elev, azim, dist, fov)

            # rgb Q
            renderer_output = tex_rgb_rendering(renderer, mesh_data, tex_rgb, elev, azim, dist, fov, shading_method='NdotV')

            img_Q_rgb = renderer_output.images[:, :, :, 0:3].permute(0, 3, 1, 2)
            pixel_uvs_rgb = renderer_output.pixel_uvs
            img_Q_rgb_list.append(img_Q_rgb.clone())
            pixel_uvs_rgb_list.append(pixel_uvs_rgb.clone())

            # latent
            renderer_output = tex_latent_rendering(renderer, diffusion_model, mesh_data, tex_latent, elev, azim, dist, fov, time_step)
            img_latent = renderer_output.images[:, :, :, 0:4].permute(0, 3, 1, 2)
            pixel_uvs_latent = renderer_output.pixel_uvs
            pixel_uvs_latent_list.append(pixel_uvs_latent)

            # latent Q
            renderer_output = tex_latent_rendering(renderer, diffusion_model, mesh_data, tex_latent, elev, azim, dist, fov, time_step, shading_method='NdotV')
            img_Q_latent = renderer_output.images[:, :, :, 0:4].permute(0, 3, 1, 2)
            img_Q_latent_list.append(img_Q_latent)

            # denoise latent
            img_latent_model = latent_denoise(diffusion_model, img_latent, depth_map, text_embeddings, time_step, guidance_scale)
            img_latent = img_latent_model.pred_original_sample
            img_latent_list.append(img_latent)
        
        time_reverse = True
        if i == end_step - 1:
            time_reverse = False
        views_merge(tex_latent, tex_rgb, tex_Q_latent, img_latent_list, img_Q_latent_list, img_Q_rgb_list, pixel_uvs_latent_list, pixel_uvs_rgb_list, diffusion_model, time_step, epochs, learning_rate, time_reverse)


def depth_map_rendering(renderer:DiffRenderer, mesh_data, tex_latent, elev, azim, dist, fov):
    '''
    depth map: [1, 3, 64, 64], (-1, 1)
    '''
    renderer.rasterization_setting(image_size=64)
    renderer.camera_setting(dist=dist, elev=elev, azim=azim, fov=fov)
    background_map = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device=tex_latent.device)

    image = renderer.rendering(mesh_data=mesh_data, diff_tex=tex_latent, 
                                shading_method='depth', background_map=background_map).images
    depth_map = (image[:, :, :, 0] * 2 - 1.0).unsqueeze(-1).permute(0, 3, 1, 2)
    return depth_map

def tex_rgb_rendering(renderer:DiffRenderer, mesh_data, tex_rgb, elev, azim, dist, fov, shading_method='no_light'):
    '''
    renderer_output:
        images: [1, 512, 512, 4], (0, 1)
        pixel_uvs: [N, 2]
    '''
    renderer.rasterization_setting(image_size=512)
    renderer.camera_setting(dist=dist, elev=elev, azim=azim, fov=fov)

    background_map = torch.ones((1, 3, 512, 512), dtype=torch.float32, device=device)

    renderer_output = renderer.rendering(mesh_data=mesh_data, diff_tex=tex_rgb,
                                                shading_method=shading_method,  
                                                background_map=background_map)

    return renderer_output

def tex_latent_rendering(renderer:DiffRenderer, diffusion_model:StableDiffusionDepth, mesh_data, tex_latent, elev, azim, dist, fov, time_step, shading_method='no_light'):
    '''
    renderer_output:
        images: [1, 64, 64, 5], (-1, 1)
        pixel_uvs: [N, 2]
    '''
    renderer.rasterization_setting(image_size=64)
    renderer.camera_setting(dist=dist, elev=elev, azim=azim, fov=fov)

    background_map = torch.zeros((1, 4, 64, 64), dtype=torch.float32, device=device)
    background_map = diffusion_model.add_noise(background_map, time_step)

    renderer_output = renderer.rendering(mesh_data=mesh_data, diff_tex=tex_latent,
                                                shading_method=shading_method,
                                                background_map=background_map)

    return renderer_output

def latent_denoise(diffusion_model:StableDiffusionDepth, img_latent, depth_map, text_embeddings, time_step, guidance_scale, eta=1.0, tau=1.0):
    '''
    img_latent_model:
        prev_sample: [1, 4, 64, 64], the latent image denoised one step
        pred_original_sample: [1, 4, 64, 64], the latent image of prediction of original
    '''
    img_latent_model = diffusion_model.latents_denoise_step(latents=img_latent, depth_map=depth_map, 
                                                            text_embeddings=text_embeddings, t=time_step, 
                                                            guidance_scale=guidance_scale, eta=eta, tau=tau,
                                                            prediction_output=True)
    
    return img_latent_model

def views_merge(tex_latent:Texture, tex_rgb:Texture, tex_Q_latent:Texture, img_latent_list, img_Q_latent_list, img_Q_rgb_list, pixel_uvs_latent_list, pixel_uvs_rgb_list, diffusion_model:StableDiffusionDepth, timestep, epochs, learning_rate, time_reverse=True):
    start_t = time.time()
    latents_noisy = tex_latent.texture.detach().clone().unsqueeze(0).permute(0, 3, 1, 2)

    tex_W_rgb = Texture(size=tex_rgb.size, device=tex_rgb.device, is_latent=False)
    tex_W_rgb.set_indentical_value(0.0)

    # update tex_W_rgb and tex_rgb
    for i in range(len(img_latent_list)):
        img_Q_rgb = img_Q_rgb_list[i].permute(0, 2, 3, 1).reshape(-1, 3).detach()
        img_latent = img_latent_list[i].detach().clone()
        img_rgb = diffusion_model.latents_decoding(img_latent).permute(0, 2, 3, 1).reshape(-1, 3)
        pixel_uvs_rgb = pixel_uvs_rgb_list[i]

        img_W_rgb = Q_weight_mapping(img_Q_rgb)
        img_W_rgb_temp = tex_W_rgb.texel_fetch(pixel_uvs_rgb)
        img_rgb_temp = tex_rgb.texel_fetch(pixel_uvs_rgb)
        tex_rgb_update(tex_rgb, tex_W_rgb, img_rgb, img_rgb_temp, img_W_rgb, img_W_rgb_temp, pixel_uvs_rgb)

    # update tex_latent
    for i in range(len(img_latent_list)):
        img_latent = img_latent_list[i].detach()
        pixel_uvs_rgb = pixel_uvs_rgb_list[i]

        rgb_gt = tex_rgb.texel_fetch(pixel_uvs_rgb).detach()

        # test for direct encoding, rather than optimization
        if epochs == 0:
            rgb_gt = rgb_gt.reshape(512, 512, 3).unsqueeze(0).permute(0, 3, 1, 2)
            # test
            utils.save_img_tensor(rgb_gt, f'Results/img_rgb_pred{timestep}_{i}.png')
            #######
            img_latent = diffusion_model.img_encoding(rgb_gt)
        ##############
        else:
            # optimize latent code
            img_latent.requires_grad = True

            optimizer = torch.optim.AdamW([img_latent], lr=learning_rate)

            #log_period = 5
            for epoch in range(epochs):
                optimizer.zero_grad()

                img_rgb = diffusion_model.latents_decoding_grad(img_latent).permute(0, 2, 3, 1).reshape(-1, 3)
                loss = torch.nn.MSELoss()(img_rgb, rgb_gt)
                loss.backward()
                optimizer.step()

                #if (epoch + 1) % log_period == 0:
                #    print(f'epoch: {epoch+1}, loss = {loss}')

        img_Q_latent = img_Q_latent_list[i].permute(0, 2, 3, 1).reshape(-1, 4)
        pixel_uvs_latent = pixel_uvs_latent_list[i]
        img_Q_latent_temp = tex_Q_latent.texel_fetch(pixel_uvs_latent)
        img_latent_temp = tex_latent.texel_fetch(pixel_uvs_latent)
        img_latent.requires_grad = False
        img_latent = img_latent.permute(0, 2, 3, 1).reshape(-1, 4)
        img_latent[img_Q_latent < img_Q_latent_temp] = img_latent_temp[img_Q_latent < img_Q_latent_temp]
        img_Q_latent[img_Q_latent < img_Q_latent_temp] = img_Q_latent_temp[img_Q_latent < img_Q_latent_temp]

        tex_latent.texel_set(pixel_uvs_latent, img_latent)
        tex_Q_latent.texel_set(pixel_uvs_latent, img_Q_latent)

        # texel add noise
        if time_reverse:
            latents_pred = tex_latent.texture.unsqueeze(0).permute(0, 3, 1, 2)

            latents = diffusion_model.custom_ddim_step(latents_pred, timestep, latents_noisy)
            latents = latents.permute(0, 2, 3, 1).squeeze(0)
            tex_latent.texture = latents

    end_t = time.time()
    #print(f'fusion process time cost: {end_t - start_t} s.')

def Q_weight_mapping(Q, T=0, p=6):
    # Q threshold
    W = torch.zeros_like(Q)
    W[Q >= T] = torch.pow(Q[Q >= T], p)

    return W

def tex_rgb_update(tex_rgb, tex_W_rgb, img_rgb, img_rgb_temp,  img_W_rgb, img_W_rgb_temp, pixel_uvs_rgb):
    img_rgb = (img_rgb_temp * (img_W_rgb_temp + 0.000000001) + img_rgb * img_W_rgb) / (img_W_rgb + img_W_rgb_temp + 0.000000001)

    tex_W_rgb.texel_set(pixel_uvs_rgb, (img_W_rgb_temp + img_W_rgb))
    tex_rgb.texel_set(pixel_uvs_rgb, img_rgb)

    return img_W_rgb

def tex_export(mesh_data, tex_rgb, camera_params, device=device, save_dir=None):
    renderer = DiffRenderer(device=device)

    camera_num = camera_params.camera_num
    dists = camera_params.dists
    elevs = camera_params.elevs
    azims = camera_params.azims
    fov = camera_params.fov

    gt_img_list = []
    for n in range(camera_num):
        dist = dists[n]
        elev = elevs[n]
        azim = azims[n]
        img_tensor = tex_rgb_rendering(renderer, mesh_data, tex_rgb, elev, azim, dist, fov, shading_method='no_light').images[:, :, :, 0:3].permute(0, 3, 1, 2)
        utils.save_img_tensor(img_tensor, save_dir + f'/rgb_fusion_gt_{n}.png')
        gt_img_list.append(img_tensor.detach())

    gt_imgs = torch.cat(gt_img_list, dim=0)
    mlp_tex = NeuralTextureField(width=32, depth=2, input_dim=2, device=device, expected_tex_size=tex_rgb.size[0])

    # learning configuration
    epochs = 500
    learning_rate = 0.1
    optimizer = torch.optim.AdamW(mlp_tex.parameters(), lr=learning_rate)
    log_update_period = 500
    start_t = time.time()
    total_loss = 0

    mlp_tex.sampling_disturb = True

    for epoch in range(epochs):
        optimizer.zero_grad()
        mlp_img_list = []
        for n in range(camera_num):
            dist = dists[n]
            elev = elevs[n]
            azim = azims[n]
            renderer.camera_setting(dist=dist, elev=elev, azim=azim, fov=fov)
            img_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=mlp_tex, shading_method='no_light').images[:, :, :, 0:3].permute(0, 3, 1, 2)
            mlp_img_list.append(img_tensor)
            mlp_imgs = torch.cat(mlp_img_list, dim=0)
        loss = nn.MSELoss()(gt_imgs, mlp_imgs)

        loss.backward()
        optimizer.step()

        total_loss += loss

        # log update
        if (epoch + 1) % log_update_period == 0:
            torch.cuda.synchronize()
            end_t = time.time()
            print(f"[INFO] Epoch {epoch+1}, takes {(end_t - start_t):.4f} s. Loss: {total_loss/log_update_period}")

            total_loss = 0
            start_t = end_t

    if save_dir is not None:
        tex_img = ((tex_rgb.texture+1.0)/2.0).unsqueeze(0).permute(0, 3, 1, 2)
        utils.save_img_tensor(tex_img, save_dir + '/tex_result.png')