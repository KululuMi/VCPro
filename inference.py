from diffusers import StableDiffusionPipeline
import torch
import glob
import numpy as np
import torch
import os
import random
# import cv2
# from PIL import Image
torch.cuda.set_device(0) 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
setup_seed(2222)
# input_path = "/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/Anti-DB/0/*"
# input_path = "/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/MIST/8/face/n004771"  # mist
# input_path = "/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs_db/MIST_VGGFACE2_n002880_face_8_100/object_DREAMBOOTH/checkpoint-1000"  # anti-db
# input_path = "/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/baseline_*/8/face/n003288"
paths=[#"/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/anti-db/8/*",
    #    "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/clean/8/*",
    #    "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/mist/8/*",
    #    "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/metacloak/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT_frequency/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/REFIT_frequency_attack/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_jpeg/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_gaussion_blur/8/*",
    
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_transfer_2_1/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_transfer_1_5/8/*"
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/PE_VGGFACE2_SPLIT/REFIT_frequency_attack/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/16/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/4/*",
    # # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/anti-db_VGG/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_womask/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_advmask/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_advmaskwithours/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_wopull/8/*",
    
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_5sense/8/*"
    
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_mix25/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_mix75/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_mix50/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/Clean-dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/Metacloak_dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/mist_dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/photoguard_imageencoder_dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/photpguard_imageencoder/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_4_60/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_8_60/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_16_60/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_20_60/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_differentsks/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/anti-db_dataset3/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_0_5/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_0_05/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_1/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_0_01/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/Photoguard_CelebA/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-_CelebA/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_0_1_16_60/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/anti-db_dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/anti-db_vgg/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/anti-db/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/metacloak_celeba/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/metacloak_dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/metacloak_vgg/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/Mist/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/Mist_dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/mist_vgg/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/photoguard_celeba/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/photoguard_dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/photoguard_vgg/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-_vgg/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-CelebA/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/SDS-_dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT-CelebA/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT-dataset3/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/anti-db/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/REFIT_frequency_attack/8/*"
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT/48/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/4/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/16/*"
    # # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_c_0_0005/4/*"
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_01/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_005/8/*",
    # "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_015/8/*",
    "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_020/8/*"
    ]
for i in paths:
    input_path = i
    
    # input_path="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/PE_VGGFACE2/metacloak/8/*"

    for filename in glob.glob(input_path):
        ti_path = os.path.join(filename,"learned_embeds-steps-3000.safetensors")
        save_path = os.path.join(filename,"images")
        save_path2 = os.path.join(filename,"images2")
        if not os.path.exists(save_path) and   os.path.exists(ti_path):
            pipeline = StableDiffusionPipeline.from_pretrained("/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,).to("cuda")
            # pipeline = StableDiffusionPipeline.from_pretrained("/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/stable-diffusion-2-1-base", torch_dtype=torch.float16, safety_checker=None,).to("cuda")

            # prompt = "a photo of <sks> person."
            prompt = "a photo of <sks>."
            # prompt = "I stood at the edge of the platform and saw a train on the platform, with no end in sight. I can also see A sign pointing in both directions of the train, with  A and B on each side."
            pipeline.load_textual_inversion(ti_path)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(save_path2)

            
        # 生成50张不同的图像
            for i in range(1, 51):
                # 生成图像
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                image_save_path = os.path.join(filename,f"images/a_photo_of_<sks>_{i}.png")
                
                print(image_save_path)
                # 保存图像，文件名按照序号命名
                image.save(image_save_path)
                # image.save(f"test_{i}.png")
            prompt = "a photo of <t@t>."
            for i in range(1, 51):
                # 生成图像
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                image_save_path = os.path.join(filename,f"images2/a_photo_of_<t@t>_{i}.png")
                
                print(image_save_path)
                # 保存图像，文件名按照序号命名
                image.save(image_save_path)