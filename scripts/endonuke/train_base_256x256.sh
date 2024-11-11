export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

accelerate launch --config_file scripts/endonuke/train_base_256x256.yaml \
    train.py \
    --root_dir /Dataset/endonuke/splits \
    --caption_list_dir data/caption_files/endonuke \
    --test_caption_files data/eval_samples/endonuke/endonuke_caption_57.txt \
    --dataset endonuke \
    --num_classes 4 \
    --exp_name base_256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 7 \
    --batch_size 7 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type endonuke \
    --split train \
    --fp16 ${@:1} \
    --num_workers 0 \
    --no_condition_on_text \
    --resume checkpoints/endonuke/joint/base_256x256/240213_064240_base_256x256/checkpoint.120000.pt

    # --noise_schedules linear linear --noise_schedules_lbl linear linear \
