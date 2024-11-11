device=$1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

python test.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/lizard/joint/base_256x256/231108_092447_base_256x256/checkpoint.300000.pt \
    --sample_timesteps 1000 \
    --start_sample_idx 552 \
    --num_samples=24 \
    --test_batch_size=1 \
    --dataset lizard \
    --num_classes 7 \
    --save_path=results/lizard/256x256_4/base.png \
    --save_img_path=outputs/lizard/256x256_1000steps/ \
    --test_captions None

# ${@:1}
