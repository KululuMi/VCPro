# -----------------------------------------------------------------------
# Copyright (c) 2023 Yixin Liu Lehigh University
# All rights reserved.
#
# This file is part of the MetaCloak project. Please cite our paper if our codebase contribute to your project. 
# -----------------------------------------------------------------------
import sys
sys.path.append('/data2/mixiaoyue/mixiaoyue/MetaCloak')
from typing import Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from LIQE.LIQE import LIQE
import tensorflow as tf
import numpy as np
import torch
from piq import CLIPIQA, BRISQUELoss
import json
clipiqa = CLIPIQA()
brisque = BRISQUELoss()
ckpt = './LIQE/checkpoints/LIQE.pt'
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32")
lieq_model = LIQE(ckpt, device = 'cuda' if torch.cuda.is_available() else 'cpu')
from robust_facecloak.generic.modi_deepface import find_without_savepkl
from deepface import DeepFace
import glob
import os 

def loop_to_get_overall_score(gen_image_dir, clean_ref_dir="", func_get_score_of_one_image=None, type_name="face"):
    files_db_gen = glob.glob(os.path.join(gen_image_dir, "*.png"))
    files_db_gen += glob.glob(os.path.join(gen_image_dir, "*.jpg"))
    scores = []
    assert len(files_db_gen) > 0
    for i in range(len(files_db_gen)):
        gen_i = files_db_gen[i]
        score = func_get_score_of_one_image(gen_i, clean_ref_dir, type_name=type_name)
        scores.append(score)
    # filter out nan and np.inf 
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    scores = scores[~np.isinf(scores)]
    scores = scores[~np.isinf(-scores)]
    return np.mean(scores)

class ScoreEval():
    def __init__(self, func_get_score_of_one_image=lambda image_dir, clean_ref_dir, type_name="face": 0):
        self.func_get_score_of_one_image = func_get_score_of_one_image
    
    def __loop_to_get_overall_score__(self, gen_image_dir, clean_ref_db=None, type_name="face"):
        files_db_gen = glob.glob(os.path.join(gen_image_dir, "*.png"))
        files_db_gen += glob.glob(os.path.join(gen_image_dir, "*.jpg"))
        scores = []
        if len(files_db_gen) == 0 : import pdb;pdb.set_trace()
        assert len(files_db_gen) > 0
        for i in range(len(files_db_gen)):
            gen_i = files_db_gen[i]
            score = self.func_get_score_of_one_image(gen_i, clean_ref_db, type_name=type_name)
            scores.append(score)
        # filter out nan and np.inf 
        scores = np.array(scores)
        scores = scores[~np.isnan(scores)]
        scores = scores[~np.isinf(scores)]
        scores = scores[~np.isinf(-scores)]
        # return np.mean(scores)
        return scores
    
    def __call__(self, gen_image_dir, clean_ref_db=None, type_name="face"):
        return self.__loop_to_get_overall_score__( gen_image_dir, clean_ref_db, type_name=type_name)


def CLIPIQA_get_score(gen_i, clean_ref_db=None, type_name="face"):
    from PIL import Image
    PIL_image = Image.open(gen_i).convert("RGB")
    from torchvision import transforms
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    with torch.no_grad():
        score = clipiqa(trans(PIL_image).unsqueeze(0)).item()
    return score
CLIP_IQA_Scorer = ScoreEval(func_get_score_of_one_image=CLIPIQA_get_score)


def LIQE_get_quality_score(gen_i, clean_ref_db=None, type_name="face"):
    img = Image.open(gen_i).convert('RGB')
    from torchvision.transforms import ToTensor
    img = ToTensor()(img).unsqueeze(0)
    q1, s1, d1 = lieq_model(img)
    return q1.item()
LIQE_Quality_Scorer = ScoreEval(func_get_score_of_one_image=LIQE_get_quality_score)

def IMS_CLIP_get_score(gen_i, clean_ref_db, type_name="face"):
    import torch
    img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    ref_pkl_path = os.path.join(clean_ref_db, "ref_mean_clip_vit_32.pkl")
    ref_representation_mean = None
    if os.path.exists(ref_pkl_path):
        ref_representation_mean= torch.load(ref_pkl_path)
    else:
        ref_images = glob.glob(os.path.join(clean_ref_db, "*.png"))
        ref_images += glob.glob(os.path.join(clean_ref_db, "*.jpg"))
        ref_representation_mean = 0.0
        for ref_image in ref_images:
            ref_image = Image.open(ref_image).convert('RGB')
            ref_image = clip_preprocess(ref_image).unsqueeze(0).to('cuda')
            with torch.no_grad():
                ref_representation_mean += clip_model.encode_image(ref_image).cpu()
        if len(ref_images)<=0:
            import pdb;pdb.set_trace()
        ref_representation_mean /= len(ref_images)
        ref_representation_mean = ref_representation_mean / ref_representation_mean.norm(dim=-1, keepdim=True)
        torch.save(ref_representation_mean, ref_pkl_path)
    
    # calculate cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    cosine_similarity = (ref_representation_mean.cpu().numpy() * image_features.cpu().numpy()).sum().mean()
    return cosine_similarity

IMS_CLIP_Scorer = ScoreEval(func_get_score_of_one_image=IMS_CLIP_get_score)

def CLIP_Face_get_score(gen_i, clean_ref_db=None, type_name="face"):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["good face", 'bad face']).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_Face_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_Face_get_score)

def CLIP_IQAC_get_score(gen_i, clean_ref_db=None, type_name="face"):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["a good photo of " + type_name, "a bad photo of " + type_name]).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_IQAC_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_IQAC_get_score)

def CLIP_zero_short_classification_get_score(gen_i, clean_ref_db=None, type_name="face"):
    import torch
    gen_img = Image.open(gen_i).convert('RGB')
    image = clip_preprocess(gen_img).unsqueeze(0).to('cuda')
    text = clip.tokenize(["a picture of " + type_name, "a picture of non-" + type_name]).to('cuda')
    similarity_matrix = None
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity_matrix[0][0].item() - similarity_matrix[0][1].item()

CLIP_zero_short_classification_Scorer = ScoreEval(func_get_score_of_one_image=CLIP_zero_short_classification_get_score)

def IMS_get_score(gen_i, clean_ref_db, type_name="face", distance_metric="cosine", model_name="VGG-Face"):
    dfs = find_without_savepkl(img_path = gen_i, db_path = clean_ref_db, enforce_detection=False, distance_metric=distance_metric, model_name=model_name, )
    all_scores = dfs[0][f'{model_name}_{distance_metric}'].values
    import numpy as np
    all_scores = all_scores[~np.isnan(all_scores)]
    dis = 0
    if len(all_scores)==0:
        dis = 2
    else:
        dis = np.mean(all_scores)
    return 1-dis

IMS_Face_Scorer = ScoreEval(func_get_score_of_one_image=IMS_get_score)

def FDSR_get_score(gen_i, clean_ref_db=None, model='retinaface', type_name="face"):
    face_obj = DeepFace.extract_faces(img_path = gen_i, target_size = (224,224), detector_backend = model, enforce_detection = False, )
    score = 0
    for i in range(len(face_obj)):
        score += face_obj[i]['confidence']
    return score / len(face_obj)

FDSR_Scorer = ScoreEval(func_get_score_of_one_image=FDSR_get_score)

def get_score(image_dir, clean_ref_dir=None, type_name="person", ):
    
    if type_name == "person":
        type_name = "face"
        
    result_dict = {}
    if type_name == "face":
        result_dict['SDS'] = FDSR_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
        result_dict['CLIP_Face_IQA'] = CLIP_Face_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    else:
        CLIP_zero_short_classification_score = CLIP_zero_short_classification_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
        result_dict['SDS'] = CLIP_zero_short_classification_score
         
    result_dict['CLIPIQA'] = CLIP_IQA_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    result_dict['LIQE_Quality'] = LIQE_Quality_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    result_dict['IMS_CLIP_ViT-B/32'] = IMS_CLIP_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    result_dict['CLIP_IQAC'] = CLIP_IQAC_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
    
    if type_name == "face":
        result_dict[f"IMS_VGG-Face_cosine"] = IMS_Face_Scorer(gen_image_dir=image_dir, clean_ref_db=clean_ref_dir, type_name=type_name)
                
    return result_dict
def ndarray_to_list(obj):
    """
    递归转换对象中的所有ndarray为list。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将ndarray转换为list
    elif isinstance(obj, dict):
        return {key: ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(element) for element in obj]
    else:
        return obj

def save_scores(score, key, base_dir='./'):
    os.makedirs(base_dir, exist_ok=True)
    # 将score中的ndarray转换为list\
    score_converted = ndarray_to_list(score)
    
    # 构建存储的文件名
    filename = f"{key}_scores.json"
    filepath = os.path.join(base_dir, filename)
    # 以JSON格式存储分数
    with open(filepath, 'w') as f:
        json.dump(score_converted, f, indent=4)
    print(f"Scores saved to {filepath}")

if __name__ == "__main__":
    image_paths=[
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_01/8/*/images",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_005/8/*/images",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_015/8/*/images",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_020/8/*/images"
                ]
    save_names=[
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/evaluations/results/vgg/refit_alpha_0_01/ti",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/evaluations/results/vgg/refit_alpha_0_005/ti",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/evaluations/results/vgg/refit_alpha_0_015/ti",
                "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/evaluations/results/vgg/refit_alpha_0_020/ti",
                ]
    
    for i in range(len(image_paths)):
        # i = i*2
        file_path=image_paths[i]
        base_path=save_names[i]
        # import pdb;pdb.set_trace()
        
        import glob
        for filename in glob.glob(file_path):
            print(filename)
            # adv_images = filename
            # clean_images = filename.replace("noise-ckpt/12","image_before_addding_noise_2")
            image_dir = filename
            # clean_ref_dir_template = "/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/8/{}/ADVERSARIAL/image_clean/{}"
            # if i==0: 
            #     matched_part = filename.split('/')[-4]
            # else: 
            # clean_ref_dir_template = "/ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2_only_images/{}"
            if i<12:
                # clean_ref_dir_template="/ssd/ssd4/mixiaoyue/diffusions/Cele-VGG-MetaCloak/CelebA-HQ-clean/{}/images"
                clean_ref_dir_template = "/ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2_only_images/{}"
                # clean_ref_dir_template="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/4/Ours_CelebA_{}_cw_mask_maskweight_4_720/CelebA_{}/image_before_addding_noise_2"
            elif i<24:
                clean_ref_dir_template = "/ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2_only_images/{}"
            else:
                clean_ref_dir_template = "/ssd/ssd4/mixiaoyue/diffusions/dataset/natural_landscapes/{}"
            # import pdb;pdb.set_trace()
            if i<24:
                # matched_part = filename.split('/')[-5].split('_')[2]
                if i %2==1:
                #     # import pdb;pdb.set_trace()
                    matched_part = filename.split('/')[-2]
                    # matched_part = filename.split('/')[-5].split('_')[2]
                else:
                    matched_part = filename.split('/')[-2]
                    # matched_part = filename.split('/')[-5].split('_')[2]
            else:
                
            # if i%2==1:
                if 'Fishermen\'s_Wharf'in filename  or 'Fishermens_Wharf' in filename:
                    matched_part= 'Fishermen\'s_Wharf'
                elif "Heart-shaped_island" in filename:
                    matched_part="Heart-shaped_island"
                elif "Statue_of_liberty" in filename:
                    matched_part="Statue_of_liberty"
                elif "Heart_lake" in filename:
                    matched_part = "Heart_lake"
                elif "Sydney_opera_house" in filename:
                    matched_part="Sydney_opera_house"
                elif "thinker" in filename:
                    matched_part="thinker"
                else:
                    if i %2==1:
                    # import pdb;pdb.set_trace()
                        matched_part = filename.split('/')[-5].split('_')[2]
                    else:
                        matched_part = filename.split('/')[-2]

            clean_ref_dir = clean_ref_dir_template.format(matched_part,matched_part)
            if os.path.exists(os.path.join(base_path, f"{matched_part}_scores.json")):
                print(os.path.join(base_path, f"{matched_part}_scores.json")+" has exits, pass....")
                continue

            score = get_score(image_dir, clean_ref_dir, type_name="person" )
            # print(score)
            
            save_scores(score, matched_part, base_dir=base_path)
