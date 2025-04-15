import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import os
from pytorch_fid import fid_score
import torch
from torchvision import datasets, transforms
import torchvision
import random
import lpips
torch.cuda.set_device(0) 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# setup_seed(20)
setup_seed(20)
def calculate_metrics(original_img, adversarial_img):
    # 计算 SSIM
    win_size = min(7, original_img.shape[0], original_img.shape[1])
    if win_size % 2 == 0:  # 窗口大小需要是奇数
        win_size = max(3, win_size - 1)  # 减一以确保是奇数，最小为 3
    # 计算 SSIM
    ssim_value = ssim(original_img, adversarial_img, multichannel=True, win_size=win_size, channel_axis=2)

    # 计算 PSNR
    psnr_value = cv2.PSNR(original_img, adversarial_img)
    
    original_img_normalized = original_img.astype(np.float32) / 255
    adversarial_img_normalized = adversarial_img.astype(np.float32) / 255

    # 计算 L2 范数
    l2_norm = np.linalg.norm(original_img_normalized - adversarial_img_normalized)

    # 计算 L1 范数
    l1_norm = np.sum(np.abs(original_img_normalized - adversarial_img_normalized))

    # 计算 Linf 范数
    linf_norm = np.max(np.abs(original_img_normalized - adversarial_img_normalized))

    return ssim_value, psnr_value, l2_norm, l1_norm, linf_norm

def process_folders(original_folder, adversarial_folder):
    # 初始化存储度量值的列表
    ssim_values, psnr_values, l2_norms, l1_norms, linf_norms,lpips_values = [], [], [], [], [],[]

    adversarial_files = [f for f in os.listdir(adversarial_folder) if os.path.isfile(os.path.join(adversarial_folder, f))]

    for file in adversarial_files:
        original_file = file.split("_noise_")[1]
        original_img = cv2.imread(os.path.join(original_folder, original_file))
        adversarial_img = cv2.imread(os.path.join(adversarial_folder, file))
        original_img = cv2.resize(original_img, (adversarial_img.shape[1], adversarial_img.shape[0]))

        if original_img is not None and adversarial_img is not None:
            ssim_value, psnr_value, l2_norm, l1_norm, linf_norm,lpips_value = calculate_metrics(original_img, adversarial_img)

            # 存储每个度量的值
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)
            l2_norms.append(l2_norm)
            l1_norms.append(l1_norm)
            linf_norms.append(linf_norm)
            lpips_values.append(lpips_value)

            # print(f"Metrics for {file}:")
            # print(f"SSIM: {ssim_value}, PSNR: {psnr_value}, L2: {l2_norm}, L1: {l1_norm}, Linf: {linf_norm}\n")
        else:
            print(f"Error loading images for {file}")

    # 计算并打印每个度量的均值和标准差
    print("Average and Standard Deviation of Metrics:")
    print(f"SSIM: Mean = {np.mean(ssim_values)}, Std = {np.std(ssim_values)}")
    print(f"PSNR: Mean = {np.mean(psnr_values)}, Std = {np.std(psnr_values)}")
    print(f"L2 Norm: Mean = {np.mean(l2_norms)}, Std = {np.std(l2_norms)}")
    print(f"L1 Norm: Mean = {np.mean(l1_norms)}, Std = {np.std(l1_norms)}")
    print(f"Linf Norm: Mean = {np.mean(linf_norms)}, Std = {np.std(linf_norms)}")
    print(f"LPIPS Norm: Mean = {np.mean(lpips_values)}, Std = {np.std(lpips_values)}")


def calculate_fid(folder1, folder2, device=None):
    batch_size = len(os.listdir(folder1))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的Inception-v3模型
    inception_model = torchvision.models.inception_v3(pretrained=True)

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 计算FID距离值
    fid_value = fid_score.calculate_fid_given_paths([folder1, folder2],
                                                    batch_size=1, device=device, dims=2048, num_workers=1)
    print('FID value:', fid_value)

    return fid_value

def process_folders_all(original_folder, adversarial_folder,ssim_values, psnr_values, l2_norms, l1_norms, linf_norms):
    # 初始化存储度量值的列表

    adversarial_files = [f for f in os.listdir(adversarial_folder) if os.path.isfile(os.path.join(adversarial_folder, f))]
    original_files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
    if 'Diff-Protect' in adversarial_folder:
        for file in original_files:
            original_file =  file
            adv_file=file.split(".")[0]+'_attacked.png'
            original_img = cv2.imread(os.path.join(original_folder, original_file))
            adversarial_img = cv2.imread(os.path.join(adversarial_folder, adv_file))
            try:
                original_img = cv2.resize(original_img, (adversarial_img.shape[1], adversarial_img.shape[0]))
            except:
                continue

            if original_img is not None and adversarial_img is not None:
                ssim_value, psnr_value, l2_norm, l1_norm, linf_norm = calculate_metrics(original_img, adversarial_img)
                # 存储每个度量的值
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)
                l2_norms.append(l2_norm)
                l1_norms.append(l1_norm)
                linf_norms.append(linf_norm)

                print(f"Metrics for {file}:")
                print(f"SSIM: {ssim_value}, PSNR: {psnr_value}, L2: {l2_norm}, L1: {l1_norm}, Linf: {linf_norm}\n")
            else:
                print(f"Error loading images for {file}")
    else:

        for file in adversarial_files:
            if 'MetaCloak' in adversarial_folder:
                original_file = file.split("noisy_")[1]
            else:
                original_file = file.split("_noise_")[1]
            original_img = cv2.imread(os.path.join(original_folder, original_file))
            adversarial_img = cv2.imread(os.path.join(adversarial_folder, file))
            original_img = cv2.resize(original_img, (adversarial_img.shape[1], adversarial_img.shape[0]))

            if original_img is not None and adversarial_img is not None:
                ssim_value, psnr_value, l2_norm, l1_norm, linf_norm = calculate_metrics(original_img, adversarial_img)
                # 存储每个度量的值
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)
                l2_norms.append(l2_norm)
                l1_norms.append(l1_norm)
                linf_norms.append(linf_norm)

                print(f"Metrics for {file}:")
                print(f"SSIM: {ssim_value}, PSNR: {psnr_value}, L2: {l2_norm}, L1: {l1_norm}, Linf: {linf_norm}\n")
            else:
                print(f"Error loading images for {file}")
    return ssim_values, psnr_values, l2_norms, l1_norms, linf_norms

images_path="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/8/Ours_VGGFACE2_*_frequency_attack_0_1_10000_c0_1_alpha_0_020/PE_VGGFACE2_*/noise-ckpt/1"
import glob
ssim_values, psnr_values, l2_norms, l1_norms, linf_norms,fids = [], [], [], [], [], []
# 使用glob.glob()遍历所有匹配的文件
setup_seed(20)
for filename in glob.glob(images_path):
    print(filename)
    matched_part = filename.split('/')[-4].split('_')[2]
    if matched_part!="100":
        adv_images = filename
        clean_images = filename.replace("noise-ckpt/12","image_before_addding_noise_2/images")
        print(clean_images)
        ssim_values, psnr_values, l2_norms, l1_norms, linf_norms = process_folders_all(clean_images,adv_images,ssim_values, psnr_values, l2_norms, l1_norms, linf_norms)
        fid = calculate_fid(clean_images, adv_images)
        fids.append(fid)
# 计算并打印每个度量的均值和标准差
print("Average and Standard Deviation of Metrics:")
print(len(ssim_values))
print(f"SSIM: Mean = {np.mean(ssim_values)}, Std = {np.std(ssim_values)}")
print(f"PSNR: Mean = {np.mean(psnr_values)}, Std = {np.std(psnr_values)}")
print(f"FID: Mean = {np.mean(fids)}, Std = {np.std(fids)}")