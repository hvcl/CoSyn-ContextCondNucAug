export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

accelerate launch --config_file scripts/pannuke/train_base_256x256.yaml \
    train.py \
    --root_dir /Dataset/pannuke/splits \
    --caption_list_dir data/caption_files/pannuke \
    --test_caption_files data/eval_samples/pannuke/pannuke_caption_57.txt \
    --dataset pannuke \
    --num_classes 6 \
    --exp_name base_256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 10000 \
    --save_every 10000 \
    --max_batch_size 7 \
    --batch_size 7 \
    --checkpoint_dir checkpoints \
    --test_batch_size 4 \
    --augmentation_type pannuke \
    --split train \
    --fp16 ${@:1} \
    --num_workers 0 \
    --no_condition_on_text

    # --noise_schedules linear linear --noise_schedules_lbl linear linear \