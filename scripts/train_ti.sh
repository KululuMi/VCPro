export MODEL_NAME="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
export EPS=8

for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
    if [ -d "$DIR" ]; then
      FOLDER_NAME=$(basename "$DIR")
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