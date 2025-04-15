export MODEL_PATH="/ssd/ssd4/mixiaoyue/diffusions/stable-diffusion-v1-4"
export CLASS_DIR="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/data/class-person"
export CLEAN_TRAIN_DIR="/ssd/ssd4/mixiaoyue/diffusions/dataset/person/person"
export REF_MODEL_PATH="/ssd/ssd4/mixiaoyue/Anti-DreamBooth/outputs/Ours/person"
export EPS=8
for DIR in /ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/*; do
    if [ -d "$DIR" ]; then
      FOLDER_NAME=$(basename "$DIR")
      # 设置实验名称和输出目录
      export EXPERIMENT_NAME="Ours_VGGFACE2_${FOLDER_NAME}_frequency_attack_0_1_10000_c0_1_alpha_0_005"
      export OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/$EPS/$EXPERIMENT_NAME/PE_VGGFACE2_$FOLDER_NAME"

      # 设置CLEAN_ADV_DIR 和 MASK_ADV_DIR
      export CLEAN_ADV_DIR="/ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2/$FOLDER_NAME"
      export MASK_ADV_DIR="/ssd/ssd4/mixiaoyue/diffusions/dataset/PE_VGGFACE2_mask/$FOLDER_NAME"
      
      if [ -d "$OUTPUT_DIR/noise-ckpt/12" ]; then
          echo "Directory $OUTPUT_DIR already exists, skipping..."
          continue  # 跳过当前迭代
      fi
      mkdir -p $OUTPUT_DIR
      cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise_2

      CUDA_VISIBLE_DEVICES=0 python attacks/vcpro_attack.py \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --train_text_encoder \
        --instance_data_dir=$CLEAN_ADV_DIR \
        --mask_data_dir=$MASK_ADV_DIR \
        --output_dir=$OUTPUT_DIR \
        --instance_prompt="a photo of sks person" \
        --resolution=512 \
        --gradient_accumulation_steps=1 \
        --max_train_steps=1 \
        --max_adv_train_steps=1000 \
        --checkpointing_steps=1 \
        --center_crop \
        --pgd_alpha=1 \
        --pgd_eps=$EPS \
        --alpha=0.005 \
        --c=0.1\

      export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/1"
      export DREAMBOOTH_OUTPUT_DIR="/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/outputs/REFIT/$EPS/DB/$EXPERIMENT_NAME/person_DREAMBOOTH"
      mkdir -p $DREAMBOOTH_OUTPUT_DIR
      CUDA_VISIBLE_DEVICES=0 python train_dreambooth.py \
          --pretrained_model_name_or_path=$MODEL_PATH  \
          --enable_xformers_memory_efficient_attention \
          --train_text_encoder \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DIR \
          --output_dir=$DREAMBOOTH_OUTPUT_DIR \
          --with_prior_preservation \
          --prior_loss_weight=1.0 \
          --instance_prompt="a photo of sks person" \
          --class_prompt="a photo of person" \
          --inference_prompt="a photo of sks person;a dslr portrait of sks person;a photo of sks person eating in thre kitchen;a photo of sks person playing in the gardern" \
          --resolution=512 \
          --train_batch_size=2 \
          --gradient_accumulation_steps=1 \
          --learning_rate=5e-7 \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --num_class_images=200 \
          --max_train_steps=1000 \
          --checkpointing_steps=1000 \
          --center_crop \
          --mixed_precision=fp16 \
          --prior_generation_precision=fp16 \
          --sample_batch_size=50
    fi
done