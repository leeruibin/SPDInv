from torchvision.utils import save_image
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl

from utils.control_utils import load_image,load_512

import argparse
from typing import Union
import torch
from diffusers import DDIMScheduler
import numpy as np
from P2P import ptp_utils
from PIL import Image
import os

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
@torch.no_grad()
def MasaCtrl_inversion_and_edit(
    model_path = "CompVis/stable-diffusion-v1-4",
    out_dir: str = "./outputs/masactrl_real/",
    source_image_path: str = "./images/statue-flower.png",
    source_prompt = "photo of a statue",
    target_prompt = "photo of a statue, side view",
    scale: float = 7.5,
    masa_step: int = 4,
    masa_layer: int = 10,
    inject_uncond: str = "src",
    inject_cond: str = "src",
    prox_step: int = 0,
    prox: str = "l0",
    quantile: float = 0.6,
    npi: bool = True,
    npi_interp: float = 1,
    npi_step: int = 0,
    num_inference_steps: int = 50,

    K_round=25,
    learning_rate = 0.001,
    enable_threhold = True,
    delta_threshold=5e-6,
    offsets=(0,0,0,0),
    **kwargs
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)
    source_image = load_image(source_image_path, device)
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    prompts = [source_prompt, target_prompt]

    SPD_inversion = SourcePromptDisentanglementInversion(model, K_round=K_round, num_ddim_steps=num_inference_steps,
                                             learning_rate=learning_rate, delta_threshold=delta_threshold, enable_threhold=enable_threhold)
    (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        source_image_path, source_prompt, offsets=offsets, npi_interp=npi_interp, verbose=True)

    start_code = x_stars[-1]

    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # hijack the attention module
    editor = MutualSelfAttentionControl(masa_step, masa_layer, inject_uncond=inject_uncond, inject_cond=inject_cond)
    # editor = MutualSelfAttentionControlMaskAuto(masa_step, masa_layer, ref_token_idx=1, cur_token_idx=2)  # NOTE: replace the token idx with the corresponding index in the prompt if needed
    regiter_attention_editor_diffusers(model, editor)

    image_masactrl = model(prompts,
                           latents=start_code,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=[1, scale],
                           neg_prompt=source_prompt if npi else None,
                           prox=prox,
                           prox_step=prox_step,
                           quantile=quantile,
                           npi_interp=npi_interp,
                           npi_step=npi_step,
                           )

    filename = source_image_path.split('/')[-1]
    save_path = f"{out_dir}/{sample_count}_MasaCtrl_{filename}"
    out_image = torch.cat([source_image * 0.5 + 0.5, image_masactrl], dim=0)
    save_image(out_image, save_path)

    print("Syntheiszed images are saved in", save_path)
    print("Real image | Reconstructed image | Edited image")

def parse_args():
    parser = argparse.ArgumentParser(description="Input your image and editing prompt.")
    parser.add_argument(
        "--input",
        type=str,
        default="images/000000000001.jpg",
        # required=True,
        help="Image path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="a round cake with orange frosting on a wooden plate",
        # required=True,
        help="Source prompt",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="a square cake with orange frosting on a wooden plate",
        # required=True,
        help="Target prompt",
    )
    parser.add_argument(
        "--K_round",
        type=int,
        default=25,
        help="Optimization Round",
    )
    parser.add_argument(
        "--num_of_ddim_steps",
        type=int,
        default=50,
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=5e-6,
        help="Delta threshold",
    )
    parser.add_argument(
        "--enable_threshold",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Save editing results",
    )
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['scale'] = args.guidance_scale
    params['K_round'] = args.K_round
    params['num_inference_steps'] = args.num_of_ddim_steps
    params['learning_rate'] = args.learning_rate
    params['enable_shrehold'] = args.enable_shrehold
    params['delta_threshold'] = args.delta_threshold

    params['source_prompt'] = args.source
    params['target_prompt'] = args.target

    params['out_dir'] = args.output
    params['source_image_path'] = args.input

    MasaCtrl_inversion_and_edit(**params)

