#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import argparse
import logging
import os
from pathlib import Path
import pandas as pd

import datasets
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline,UNet2DConditionModel
from diffusers.utils import check_min_version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig
import PIL
from tqdm import tqdm
import numpy as np
import pandas as pd
from DWT import *
import torchvision.models as models
import torch.nn.functional as F
from scipy.ndimage import rotate
import math
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
logger = get_logger(__name__)
loss_data = pd.DataFrame(columns=["Iteration", "Loss_vae","Loss_unet","Loss_unet_mask"])
def normalization(x):
    return transforms.Normalize([0.5], [0.5])(x)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--mask_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=1.0 / 255,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="The value of alpha.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.01,
        help="The value of c.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(data_dir, mask_dir=None, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = torch.stack(images)
    if mask_dir!=None:
        mask_dir_list =[ os.path.basename(i) for i in list(Path(data_dir).iterdir())]
        mask_dir_list = [os.path.join(mask_dir,os.path.splitext(i)[0])+'.png' for i in mask_dir_list]
        masks = [Image.open(i) for i in mask_dir_list]
        
        for i in range(len(masks)):
            masks[i] = np.array(masks[i].resize((size,size),resample=PIL.Image.BILINEAR)).astype(np.uint8)
            # import pdb; pdb.set_trace()
            mask = masks[i]
            mask[mask<125]=0
            mask[mask>=125]=1
            masks[i] = torch.from_numpy(mask)
        masks = torch.stack(masks)
        return images, masks
    return images


def load_model(args, model_path):
    
    model_base_path="/data/share/diffusions/stable-diffusion-v1-4"#ADD BY MI
    print(model_path)
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=args.revision)

    # num_iters = 100
    # num_train_steps = 20
    # num_pgd_attack_steps = 20

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(model_base_path, subfolder="vae", revision=args.revision)

    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        print("You selected to used efficient xformers")
        print("Make sure to install the following packages before continue")
        print("pip install triton==2.0.0.dev20221031")
        print("pip install pip install xformers==0.0.17.dev461")

        unet.enable_xformers_memory_efficient_attention()

    return text_encoder, unet, tokenizer, noise_scheduler, vae

def cw_frequency_attack(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    num_steps: int,
    mask_image=None,
    loss_data=None,
    alpha=0.05,
    c=0.1
):
    """Return new perturbed data"""
    tau=0.9
    DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
    MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
    ABORT_EARLY = True      # abort gradient descent upon first valid solution
    INITIAL_CONST = 1e-5    # the first value of c to start at
    LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
    LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
    REDUCE_CONST = False    # try to lower c each iteration; faster to set to false
    CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better
    
    # mask_image = None
   

    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")
    # device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    wave= 'haar'
    DWT = DWT_2D_tiny(wavename= wave).to(device)
    IDWT = IDWT_2D_tiny(wavename= wave).to(device)
    lowFre_loss = nn.SmoothL1Loss(reduction='sum').to(device)
    
    # lowFre_loss = nn.MSE

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    perturbed_images = data_tensor.detach().clone()
    perturbed_images = perturbed_images.to(device)
    base_image = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)
    base_image.requires_grad_(False)
    

    
    
    
    input_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)
    latents_clean = vae.encode(normalization(original_images).to(device, dtype=weight_dtype)).latent_dist.sample()
    latents_clean = latents_clean*vae.config.scaling_factor

    if mask_image !=None:
        target_image = original_images.detach().clone()*(1-mask_image)
        target_image.requires_grad_(False)
        original_images = original_images.to(device)
        target_latent_tensor = vae.encode(normalization(target_image).to(device, dtype=weight_dtype)).latent_dist.sample()
        target_latent_tensor = target_latent_tensor*vae.config.scaling_factor

    for step in range(num_steps):
        perturbed_images.requires_grad = True
        
        # perturbed_images = mask_drop_out*perturbed_images
        latents = vae.encode(normalization(perturbed_images).to(device, dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # N=4, C, 64, 64

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]
        

        # # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        unet.zero_grad()
        text_encoder.zero_grad()
        
        if mask_image!=None:
            mask_image = mask_image.to(device, dtype=weight_dtype)
            mask_image_unet = transforms.Resize(model_pred.shape[-1], interpolation=transforms.InterpolationMode.BILINEAR)(mask_image)
            mask_loss_weight = mask_image.reshape(-1).shape[0]/mask_image.sum() #bz=1
            mask_loss_weight = mask_loss_weight.float()
            loss1=-F.mse_loss((latents).float(), (target_latent_tensor).float(), reduction="mean")
            loss_mask = F.mse_loss((model_pred*mask_image_unet).float(), (target*mask_image_unet).float(), reduction="mean")
            loss_mask_bu = F.mse_loss((model_pred*(1-mask_image_unet)).float(), (target*(1-mask_image_unet)).float(), reduction="mean")
            loss2=mask_loss_weight*loss_mask-loss_mask_bu
            loss3 = F.mse_loss((model_pred).float(), (target).float(), reduction="mean")
            inputs_ll = DWT(original_images.to(device))
            inputs_ll = IDWT(inputs_ll)
            adv_ll = DWT(perturbed_images.to(device))
            adv_ll = IDWT(adv_ll)
            lowFre_cost = lowFre_loss(adv_ll, inputs_ll)
            loss4 = F.mse_loss((perturbed_images).float(), (original_images).float(), reduction="mean")
            loss =  (loss2-1)-c*lowFre_cost
            new_row = {"Iteration": step + 1,"lowFre_cost":lowFre_cost.item(), "Loss_vae": -loss1.item(), "Loss_unet": loss3.item(), "Loss_unet_mask": loss_mask.item(),"Loss_unet_mask_bu":loss_mask_bu.item(),"loss2":loss2.item(),"Final_loss":loss.item(),"timestep":timesteps}
            loss_data.append(new_row)
                
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        
        loss.backward(retain_graph=True)
        pgd_alpha = args.pgd_alpha/255.0
        eps = args.pgd_eps/255.0
        
        adv_images = perturbed_images + pgd_alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
        perturbed_images = torch.clamp(original_images + eta, min=0, max=+1).detach_()
        if alpha is not None:
            if timesteps[0]>=800:
                if loss_mask>=alpha:
                    return perturbed_images
    return perturbed_images

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.mask_data_dir!=None:
        perturbed_data,mask_data = load_data(
            args.instance_data_dir,
            mask_dir =args.mask_data_dir ,
            size=args.resolution,
            center_crop=args.center_crop,
        )
        mask_data.requires_grad_(False)
    else:
        perturbed_data = load_data(
            args.instance_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
        )
        mask_data=None
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)
    loss_data=[]
    

    model_paths = list(args.pretrained_model_name_or_path.split(","))
    num_models = len(model_paths)
    
    text_encoder, unet, tokenizer, noise_scheduler, vae = load_model(args, model_paths[0])
    f = (unet, text_encoder)
    alpha = args.alpha

    for i in tqdm(range(int(args.max_train_steps))):
        
        for num in range(len(perturbed_data)):
                perturbed_ = perturbed_data[num].unsqueeze(0)
                original_ = original_data[num].unsqueeze(0)
                mask = mask_data[num].unsqueeze(0)
                en_data = 0.0
                for j, model_path in enumerate(model_paths):                    
                    en_data += (
                        cw_frequency_attack(args,f,tokenizer,noise_scheduler,vae,perturbed_,original_,args.max_adv_train_steps,mask,loss_data,alpha=alpha)/ num_models
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # update
                perturbed_ = en_data
                perturbed_data[num] = perturbed_
            

        if (i + 1) % args.checkpointing_steps == 0:
            save_folder = f"{args.output_dir}/noise-ckpt/{i+1}"
            os.makedirs(save_folder, exist_ok=True)
            noised_imgs = perturbed_data.detach()
            img_names = [
                str(instance_path).split("/")[-1] for instance_path in list(Path(args.instance_data_dir).iterdir())
            ]
            for img_pixel, img_name in zip(noised_imgs, img_names):
                save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}")
                Image.fromarray(
                    (img_pixel * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)
            print(f"Saved noise at step {i+1} to {save_folder}")
    df = pd.DataFrame(loss_data)
    df.to_excel(args.output_dir+"/loss_values.xlsx", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)