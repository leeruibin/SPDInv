import os
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import random
import argparse
import torch.nn as nn
import torchvision.transforms as T
from utils.utils import txt_draw, load_512
from utils.wavelet_color_fix import wavelet_color_fix
from P2P.SPDInv_for_PNP import get_PNP_input
from PNP.pnp_utils import register_time,register_attention_control_efficient,register_conv_control_efficient,dilate
from PNP.pnp_utils import get_timesteps

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PNP(nn.Module):
    def __init__(self, model_key, n_timesteps=50, device="cuda"):
        super().__init__()
        self.device = device
        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps = n_timesteps
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self, image_path):
        # load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t, guidance_scale, noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent] + [x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']

        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self, image_path, noisy_latent, target_prompt, guidance_scale=7.5, pnp_f_t=0.8, pnp_attn_t=0.5,image_enc=None):
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]
        self.image_enc = image_enc

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps, guidance_scale, noisy_latent)

        return edited_img

    def sample_loop(self, x, guidance_scale, noisy_latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(self.scheduler.timesteps):
                x = self.denoise_step(x, t, guidance_scale, noisy_latent[-1 - i])

            decoded_latent = self.decode_latent(x)

        return decoded_latent


def PnP_inversion_and_edit(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        fix_color=True,
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)

    _, rgb_reconstruction, latent_reconstruction,ref_latent = get_PNP_input(image_path=image_path,prompt_src=prompt_src)

    edited_image = pnp.run_pnp(image_path, latent_reconstruction, prompt_tar, guidance_scale)

    transform = T.ToPILImage()
    target_image = transform(edited_image[0])
    if fix_color:
        target_image = wavelet_color_fix(target_image, Image.fromarray(image_gt))

    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255 * np.array(rgb_reconstruction[0].permute(1, 2, 0).cpu().detach())),
        np.asarray(target_image),
    ), 1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="images/000000000001.jpg")  # the editing category that needed to run
    parser.add_argument('--output', type=str, default="outputs")  # the editing category that needed to run
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--source', type=str, default="a round cake with orange frosting on a wooden plate")
    parser.add_argument('--target', type=str, default="a square cake with orange frosting on a wooden plate")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')

    # model_key = "CompVis/stable-diffusion-v1-5"
    model_key = "runwayml/stable-diffusion-v1-4"
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(args.num_ddim_steps)

    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=args.num_ddim_steps,
                                                           strength=1.0,
                                                           device=device)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))

    image_path = args.input
    prompt_src = args.source
    prompt_tar = args.target

    pnp = PNP(model_key)
    edited_image = PnP_inversion_and_edit(
        image_path=image_path,
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        guidance_scale=7.5,
    )
    filename = image_path.split('/')[-1]
    edited_image.save(f"{output_dir}/{sample_count}_P2P_{filename}")


