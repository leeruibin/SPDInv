from typing import Optional, Union, List
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

import ptp_utils
from PIL import Image
from scheduler_dev import DDIMSchedulerDev
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from ptp_utils import load_512


class SourcePromptDisentanglementInversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def SPDInv_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.model.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            latent_ztm1 = latent.clone().detach()
            latent = self.next_step(noise_pred, t, latent_ztm1)

            ################ SPDInv optimization steps #################
            optimal_latent = latent.clone().detach()
            optimal_latent.requires_grad = True
            optimizer = torch.optim.AdamW([optimal_latent], lr=self.lr)
            for rid in range(self.K_round):
                with torch.enable_grad():
                    optimizer.zero_grad()
                    noise_pred = self.model.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
                    pred_latent = self.next_step(noise_pred, t, latent_ztm1)
                    loss = F.mse_loss(optimal_latent, pred_latent)
                    loss.backward()
                    optimizer.step()
                    if self.enable_shrehold and loss < self.delta_threshold:
                        break
            ############### End SPDInv optimization ###################

            latent = optimal_latent.clone().detach()
            latent.requires_grad = False
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def SPD_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.SPDInv_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("Source Prompt Disentanglement inversion...")
        image_rec, ddim_latents, image_rec_latent = self.SPD_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threhold=True):

        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.K_round = K_round
        self.num_ddim_steps = num_ddim_steps
        self.lr = learning_rate
        self.delta_threshold = delta_threshold
        self.enable_shrehold = enable_threhold


# %%
@torch.no_grad()
def editing_p2p(
        model,
        prompt: List[str],
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        return_type='image',
        inference_stage=True,
        x_stars=None,
        **kwargs,
):
    batch_size = len(prompt)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    start_time = num_inference_steps
    model.scheduler.set_timesteps(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, None, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def get_PNP_input(
        image_path,
        prompt_src,
        guidance_scale=7.5,
        npi_interp=0,
        offsets=(0, 0, 0, 0),
        K_round=25,
        num_of_ddim_steps=50,
        learning_rate=0.001,
        delta_threshold=5e-6,
        enable_threshold=True,
):
    scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                 set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(
        device)

    SPD_inversion = SourcePromptDisentanglementInversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=learning_rate, delta_threshold=delta_threshold,
                                                         enable_threhold=enable_threshold)

    (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)

    z_inverted_noise_code = x_stars[-1]
    del SPD_inversion
    torch.cuda.empty_cache()

    prompts = [prompt_src]

    rgb_reconstruction, latent_reconstruction = editing_p2p(ldm_stable, prompts, latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings, x_stars=x_stars, )
    ref_latent = image_enc_latent

    return z_inverted_noise_code, rgb_reconstruction, latent_reconstruction, ref_latent


