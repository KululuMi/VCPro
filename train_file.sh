export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
export EPS=8

for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
    if [ -d "$DIR" ]; then
      FOLDER_NAME=$(basename "$DIR")
      # 设置实验名称和输出目录
        # export DATA_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/${EPS}/Ours_VGGFACE2_${FOLDER_NAME}_frequency_attack_0_1_5sensemask${EPS}_720/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/12"
        # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
        export DATA_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/${EPS}/Ours_VGGFACE2_${FOLDER_NAME}_frequency_attack_0_1_10000_c0_1_alpha_0_020/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/1"
        export OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency_alpha_0_020/${EPS}/${FOLDER_NAME}"
        if [ -d "$OUTPUT_DIR" ]; then
          echo "Directory $OUTPUT_DIR already exists, skipping..."
          continue  # 跳过当前迭代
        fi
        
        mkdir -p $OUTPUT_DIR

        CUDA_VISIBLE_DEVICES=5 python textual_inversion.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="person" \
        --placeholder_token="<sks>" --initializer_token="human" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=3000 \
        --learning_rate=5.0e-04 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --output_dir=$OUTPUT_DIR \
        --num_vectors 8
    fi
done

# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
# export EPS=4

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
#         export DATA_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/${EPS}/Ours_VGGFACE2_${FOLDER_NAME}_frequency_attack_0_1_${EPS}_720/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/12"
#         # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
#         export OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/TI_ouputs/REFIT_frequency/${EPS}/${FOLDER_NAME}"
#         if [ -d "$OUTPUT_DIR" ]; then
#           echo "Directory $OUTPUT_DIR already exists, skipping..."
#           continue  # 跳过当前迭代
#         fi
        
#         mkdir -p $OUTPUT_DIR

#         CUDA_VISIBLE_DEVICES=7 python textual_inversion.py \
#         --pretrained_model_name_or_path=$MODEL_NAME \
#         --train_data_dir=$DATA_DIR \
#         --learnable_property="person" \
#         --placeholder_token="<sks>" --initializer_token="human" \
#         --resolution=512 \
#         --train_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=3000 \
#         --learning_rate=5.0e-04 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --output_dir=$OUTPUT_DIR \
#         --num_vectors 8
#     fi
# done

#Clean_TI_Test
# sleep 30000
# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
# export EPS=0

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/TI_PEours/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
#         export DATA_DIR="/ssd/ssd4/mixiaoyue/diffusions/dataset/TI_PEours/${FOLDER_NAME}"
#         export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/Clean/${EPS}/${FOLDER_NAME}"
#         if [ -d "$OUTPUT_DIR" ]; then
#           echo "文件夹已存在: $OUTPUT_DIR"
#         else
#           mkdir -p $OUTPUT_DIR

#           CUDA_VISIBLE_DEVICES=1 python textual_inversion.py \
#           --pretrained_model_name_or_path=$MODEL_NAME \
#           --train_data_dir=$DATA_DIR \
#           --learnable_property="object" \
#           --placeholder_token="<sks>" --initializer_token="object" \
#           --resolution=512 \
#           --train_batch_size=1 \
#           --gradient_accumulation_steps=4 \
#           --max_train_steps=3000 \
#           --learning_rate=5.0e-04 --scale_lr \
#           --lr_scheduler="constant" \
#           --lr_warmup_steps=0 \
#           --output_dir=$OUTPUT_DIR \
#           --num_vectors 8
#         fi
#     fi
# done


# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
# export EPS=4

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/TI_PEours/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
#         export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/ANTI_DB_TI_PEours_${FOLDER_NAME}_object_cw_mse_mask_maskweight_${EPS}_720/TI_PEours_${FOLDER_NAME}/noise-ckpt/12"
#         # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
#         export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/ANTI-DB/${EPS}/${FOLDER_NAME}"
#         mkdir -p $OUTPUT_DIR

#         CUDA_VISIBLE_DEVICES=6 python textual_inversion.py \
#         --pretrained_model_name_or_path=$MODEL_NAME \
#         --train_data_dir=$DATA_DIR \
#         --learnable_property="object" \
#         --placeholder_token="<sks>" --initializer_token=${FOLDER_NAME} \
#         --resolution=512 \
#         --train_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=3000 \
#         --learning_rate=5.0e-04 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --output_dir=$OUTPUT_DIR \
#         --num_vectors 8
#     fi
# done
# #Ours FACE
# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-2-1-base" #transfer attack 1.4->2.1
# export EPS=8

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
#         export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/Ours_VGGFACE2_${FOLDER_NAME}_face_cw_mse_mask_maskweight_${EPS}_720/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/12"
#         # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
#         export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/Ours_transfer2_1/${EPS}/${FOLDER_NAME}"
#         mkdir -p $OUTPUT_DIR

#         CUDA_VISIBLE_DEVICES=7 python textual_inversion.py \
#         --pretrained_model_name_or_path=$MODEL_NAME \
#         --train_data_dir=$DATA_DIR \
#         --learnable_property="object" \
#         --placeholder_token="<sks>" --initializer_token="human" \
#         --resolution=512 \
#         --train_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=3000 \
#         --learning_rate=5.0e-04 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --output_dir=$OUTPUT_DIR \
#         --num_vectors 8
#     fi
# done

# #Ours FACE_Ablation
# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4" #transfer attack 1.4->1.5
# export EPS=8

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
#         export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/Ours_VGGFACE2_${FOLDER_NAME}_face_mask_maskweight_${EPS}_720/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/12"
#         # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
#         export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/Ours_ablation_mask/${EPS}/${FOLDER_NAME}"
#         mkdir -p $OUTPUT_DIR

#         CUDA_VISIBLE_DEVICES=7 python textual_inversion.py \
#         --pretrained_model_name_or_path=$MODEL_NAME \
#         --train_data_dir=$DATA_DIR \
#         --learnable_property="object" \
#         --placeholder_token="<sks>" --initializer_token="human" \
#         --resolution=512 \
#         --train_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=3000 \
#         --learning_rate=5.0e-04 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --output_dir=$OUTPUT_DIR \
#         --num_vectors 8
#     fi
# done




# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
# export EPS=8

# export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/Ours_VGGFACE2_n003288_baseline_masknoise_8_720/PE_VGGFACE2_n003288/noise-ckpt/12"
# # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
# export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/baseline_masknoise/8/face/n003288"
# mkdir -p $OUTPUT_DIR

# CUDA_VISIBLE_DEVICES=5 python textual_inversion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token="<sks>" --initializer_token="human" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=3000 \
#   --learning_rate=5.0e-04 --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --num_vectors 8

# export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/MIST_VGGFACE2_n003288_face_8_100/PE_VGGFACE2_n003288/noise-ckpt-mask"
# # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
# export OUTPUT_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/TI_outputs/baseline_noisemask_mist/8/face/n003288"
# mkdir -p $OUTPUT_DIR

# CUDA_VISIBLE_DEVICES=5 python textual_inversion.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token="<sks>" --initializer_token="human" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=3000 \
#   --learning_rate=5.0e-04 --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --num_vectors 8

# #Ours FACE 5sense
# export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4" #transfer attack 1.4->1.5
# export EPS=8

# for DIR in /ssd/ssd4/mixiaoyue/diffusions/Cele-VGG-MetaCloak/CelebA-HQ-clean/*; do
#     if [ -d "$DIR" ]; then
#       FOLDER_NAME=$(basename "$DIR")
#       # 设置实验名称和输出目录
      
#         # export DATA_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/Ours_VGGFACE2_${FOLDER_NAME}_5sense_mask_maskweight_${EPS}_720/PE_VGGFACE2_${FOLDER_NAME}/noise-ckpt/12"
#         # export DATA_DIR="/ssd/ssd4/mixiaoyue/diffusions/Cele-VGG-MetaCloak/CelebA-HQ-clean/${FOLDER_NAME}/images"
#         # export DATA_DIR="/data/mixiaoyue/Anti-DreamBooth/outputs3/ASPL/${EPS}/${FOLDER_NAME}/ADVERSARIAL/noise-ckpt/50"
#         export DATA_DIR="/data2/mixiaoyue/mixiaoyue/MetaCloak/exp_data/gen_output/release-MetaCloak-advance_steps-2-total_trail_num-4-unroll_steps-1-interval-200-total_train_steps-1000-SD14-robust-gauK-7/dataset-CelebA-HQ-clean-r-8-model-SD14-gen_prompt-sks/${FOLDER_NAME}/noise-ckpt/final"
#         # export OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/clean/${EPS}/${FOLDER_NAME}"
#         export OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/TI_ouputs/CelebA/metacloak/${EPS}/${FOLDER_NAME}"
#         mkdir -p $OUTPUT_DIR

#         CUDA_VISIBLE_DEVICES=0 python textual_inversion.py \
#         --pretrained_model_name_or_path=$MODEL_NAME \
#         --train_data_dir=$DATA_DIR \
#         --learnable_property="object" \
#         --placeholder_token="<sks>" --initializer_token="human" \
#         --resolution=512 \
#         --train_batch_size=1 \
#         --gradient_accumulation_steps=4 \
#         --max_train_steps=3000 \
#         --learning_rate=5.0e-04 --scale_lr \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --output_dir=$OUTPUT_DIR \
#         --num_vectors 8
#     fi
# done