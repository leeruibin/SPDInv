"""
Util functions based on Diffuser framework.
"""

import torch
import numpy as np

# import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
# from torchvision.utils import save_image
# from torchvision.io import read_image

from diffusers import StableDiffusionPipeline
# from pytorch_lightning import seed_everything
from P2P.ptp_utils import slerp_tensor

class MasaCtrlPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        prox=None,
        prox_step=0,
        quantile=0.7,
        npi_interp=0,
        npi_step=0,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        
        if isinstance(guidance_scale, (tuple, list)):
            assert len(guidance_scale) == 2
            # guidance_scale_batch = torch.tensor(guidance_scale, device=DEVICE).reshape(2, 1, 1, 1)
            guidance_scale_0, guidance_scale_1 = guidance_scale[0], guidance_scale[1]
            guidance_scale = guidance_scale[1]
            do_separate_cfg = True
        else:
            # guidance_scale_batch = torch.tensor([guidance_scale], device=DEVICE).reshape(1, 1, 1, 1)
            do_separate_cfg = False

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            if npi_interp > 0:
                assert neg_prompt is not None, "Please provide negative prompt for NPI."
                null_embedding = self.tokenizer(
                    [""] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                neg_embedding = self.tokenizer(
                    [neg_prompt] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                neg_embedding = self.text_encoder(neg_embedding.input_ids.to(DEVICE))[0]
                # unconditional_embeddings = (1-npi_interp) * npi_embedding + npi_interp * null_embedding
                unconditional_embeddings = slerp_tensor(npi_interp, neg_embedding, null_embedding)
                # unconditional_embeddings = unconditional_embeddings.repeat(batch_size, 1, 1)
                unconditional_embeddings = torch.cat([neg_embedding, unconditional_embeddings], dim=0)
            else:
                unconditional_input = self.tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            if npi_step > 0:
                null_embedding = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                text_embeddings_null = torch.cat([null_embedding, text_embeddings], dim=0)
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            else:
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise
            if npi_step >= 0 and i < npi_step:
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings_null).sample
            else:
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # if guidance_scale > 1.:
            #     noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            #     noise_pred = noise_pred_uncon + guidance_scale_batch * (noise_pred_con - noise_pred_uncon)
            
            # do CFG separately for source and target
            if do_separate_cfg:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred_0 = noise_pred_uncon[:batch_size//2,...] + guidance_scale_0 * (noise_pred_con[:batch_size//2,...] - noise_pred_uncon[:batch_size//2,...])                
                score_delta = noise_pred_con[batch_size//2:,...] - noise_pred_uncon[batch_size//2:,...]
                if (i >= prox_step) and (prox == 'l0' or prox == 'l1'):
                    if quantile > 0:
                        threshold = score_delta.abs().quantile(quantile)
                    else:
                        threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                    score_delta -= score_delta.clamp(-threshold, threshold)  # hard thresholding
                    if prox == 'l1':
                        score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                        score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                noise_pred_1 = noise_pred_uncon[batch_size//2:,...] + guidance_scale_1 * score_delta
                noise_pred = torch.cat([noise_pred_0, noise_pred_1], dim=0)
            else:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            batch_size = 1
        else:
            batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
