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
from P2P.SPDInv import SourcePromptDisentanglementInversion

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    enable_threshold = True,
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
                                             learning_rate=learning_rate, delta_threshold=delta_threshold, enable_threshold=enable_threshold)

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
        help="Blended word needed for MasaCtrl",
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
    params['enable_threshold'] = args.enable_threshold
    params['delta_threshold'] = args.delta_threshold

    params['source_prompt'] = args.source
    params['target_prompt'] = args.target

    params['out_dir'] = args.output
    params['source_image_path'] = args.input

    MasaCtrl_inversion_and_edit(**params)

