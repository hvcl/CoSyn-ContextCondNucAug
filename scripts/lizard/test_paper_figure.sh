device=$1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$1

python test_paper_figure.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/lizard/joint/base_256x256/231108_092447_base_256x256/checkpoint.300000.pt \
    --sample_timesteps 1000 \
    --start_sample_idx 0 \
    --num_samples=24 \
    --test_batch_size=1 \
    --dataset lizard \
    --num_classes 7 \
    --save_path=results/lizard/paper_figure_teaser5/ \
    --save_img_path=outputs/lizard/paper_figure_teaser5/ \
    --test_captions None

# ${@:1}
