device=$1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

python test.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/endonuke/joint/base_256x256/240213_064240_base_256x256/checkpoint.300000.pt \
    --sample_timesteps 1000 \
    --start_sample_idx 171 \
    --num_samples=9 \
    --test_batch_size=1 \
    --dataset endonuke \
    --num_classes 4 \
    --save_path=results/endonuke/300000/base.png \
    --save_img_path=outputs/endonuke/300000/ \
    --test_captions None

# ${@:1}
