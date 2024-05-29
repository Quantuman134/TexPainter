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
    