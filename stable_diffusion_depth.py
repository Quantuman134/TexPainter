import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler

class StableDiffusionDepth():
    def __init__(self, library="stabilityai/stable-diffusion-2-depth", num_inference_steps=50, device='cpu'):
        self.device = device

        # load model from library
        self.vae = AutoencoderKL.from_pretrained(library, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(library, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(library, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(library, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(library, subfolder="scheduler")

        self.scheduler.set_timesteps(num_inference_steps)
        self.height = 512
        self.width = 512     

    def text_encoding(self, text_prompt=""):
        prompt = [text_prompt]
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length  = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([''], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def add_noise(self, latents, t):
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        return latents_noisy
    
    def latents_decoding(self, latents):
        with torch.no_grad():
            # range of color value [0, 1.0]
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
            images = (images + 1) / 2.0 

        return images

    def latents_decoding_grad(self, latents):
        # range of color value [0, 1.0]
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images + 1) / 2.0 

        return images

    def img_encoding(self, imgs):
        with torch.no_grad():
            # range of color value [0, 1.0]
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.sample() * 0.18215

        return latents

    def latents_denoise_step(self, latents, depth_map, text_embeddings, t, guidance_scale=7.5, eta=1.0, tau=1.0, prediction_output=False):
        # depth_map [1, 3, 64, 64], value_range [-1, 1] 

        # for classifier-free guidance, duplicate the depth map
        depth_map = torch.cat([depth_map] * 2)

        # classifier-free guidance, two latents for conditional and unconditional generate in on forward pass
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat([latent_model_input, depth_map], dim=1)

        # predict noise, and disable the gradient calculate within unet to avoid tremendous memory prossession
        with torch.no_grad():
            noise_pred =  self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # classifier-free guidance weighted summarize the conditional and unconditional noise predction
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # according to predicted noise, update the latents
        if prediction_output:
            latents_model = self.scheduler.step(noise_pred, t, latents, eta=eta, tau=tau)
            return latents_model
        else:
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta, tau=tau).prev_sample 
            return latents
        
    def custom_ddim_step(self, latents_pred, timestep, latents_noisy, eta=1.0, tau=1.0):
        noise = torch.randn_like(latents_noisy)

        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma = eta * self.scheduler._get_variance(timestep, prev_timestep) ** 0.5

        noise_pred = (latents_noisy - latents_pred * (alpha_prod_t ** 0.5)) / (beta_prod_t ** 0.5) * (1 - alpha_prod_t_prev - sigma ** 2) ** 0.5

        latents_denoised = alpha_prod_t_prev ** 0.5 * latents_pred + noise_pred + sigma * noise * tau

        return latents_denoised
    