# ![alt text](image.png) Visual-Friendly Concept Protection via Selective Adversarial Perturbations
> TLDR; we propose visual-friendly perturbation against unauthorized personalized generation with diffusion models (e.g., Textual Inversion, Dreambooth).

> This is the official implementation of the paper "Visual-Friendly Concept Protection via Selective Adversarial Perturbations". [📄Paper](https://arxiv.org/pdf/2408.08518);

## Software Dependencies
```shell
conda create -n VCPro python=3.8.18
conda activate VCPro
pip install -r requirement.txt --ignore-installed
pip install git+https://github.com/huggingface/diffusers.git
```

## Checkpoint Dependencies
- put LIQE checkpoint `https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view` to `./LIQE/checkpoints/`
- model: download sd models into `./SD/` folder
    - 2.1base https://huggingface.co/stabilityai/stable-diffusion-2-1-base
    - 1.5 https://huggingface.co/runwayml/stable-diffusion-v1-5
    - 1.4 https://huggingface.co/CompVis/stable-diffusion-v1-4 
> GPU allocation: All experiments are performed on a single NVIDIA 40GB A100 GPU.

## Data
We have experimented on these two datasets: VGGFace2 and CelebA-HQ.
Following Anti-DreamBooth, we  select 50 identities from each dataset and carefully choose a subset of 12 images for each individual based on good pose and lighting.

## Environment Setup
setup the following environment variables 
```shell
# your project root
export ADB_PROJECT_ROOT="/path/to/your/project/root"
# your conda env name
export PYTHONPATH=$PYTHONPATH$:$ADB_PROJECT_ROOT
```

## Scripts 

```shell
- scripts
    -- attack_vcpro.sh #this is for generation Perturbations
    -- train_db.sh #this is for training DreamBooth
    -- train_ti.sh #this is for training Textual Inversion
```

## Evaluation
```shell
# please modify some config in the script before running
cd evaluations
python visual_quallity.py #evaluation for perturbation visibilty
python robust_facecloak/eval_score.py
# Then foor loop all the instances and compute the metrics 
```


## Citation
If our work or codebase is useful for your research, please consider citing:
```bibtex
@article{mi2024visual,
  title={Visual-friendly concept protection via selective adversarial perturbations},
  author={Mi, Xiaoyue and Tang, Fan and Cao, Juan and Li, Peng and Liu, Yang},
  journal={arXiv preprint arXiv:2408.08518},
  year={2024}
}
```


## Acknowledgement
- [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth)
- [MetaCloak](https://github.com/liuyixin-louis/MetaCloak)
- [CLIP-IQA](https://github.com/IceClear/CLIP-IQA?tab=readme-ov-file)
- [deepface](https://github.com/serengil/deepface)
- [REM](https://github.com/fshp971/robust-unlearnable-examples)
