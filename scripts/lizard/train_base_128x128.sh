export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

accelerate launch --config_file scripts/lizard/train_base_128x128.yaml \
    train.py \
    --root_dir /Dataset/lizard/NASDM/lizard_split_norm \
    --caption_list_dir data/caption_files/lizard \
    --test_caption_files data/eval_samples/lizard/lizard_caption_57.txt \
    --dataset lizard \
    --num_classes 7 \
    --exp_name base_128x128 \
    --model_type base_128x128 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 8 \
    --batch_size 8 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type lizard \
    --split train \
    --fp16 ${@:1} \
    --num_workers 0 \
    --no_condition_on_text

    # --noise_schedules linear linear --noise_schedules_lbl linear linear \
