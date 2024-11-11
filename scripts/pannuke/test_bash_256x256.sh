device=$1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

python test.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/pannuke/joint/base_256x256/240213_064240_base_256x256/checkpoint.300000.pt \
    --sample_timesteps 1000 \
    --start_sample_idx 1100 \
    --num_samples=50 \
    --test_batch_size=1 \
    --dataset pannuke \
    --num_classes 6 \
    --save_path=results/pannuke/300000_val/base.png \
    --save_img_path=outputs/pannuke/300000_val/ \
    --test_captions None

# ${@:1}
