
GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python meta_train.py \
                --gpu $GPU \
                --folder_name 'dass' \
                --mode 'meta_train' \
                --image_size 64 \
                --meta_lr 5e-4 \
                --num_episodes 1000 \
                --mvld_frequency 20 \
                --h_dim 64 \
                --num_inner_updates 1 \
                --first_order F \
                --step_size 0.005 \
                --n_q 50 \
                --task_lr T \
                --pr_type 'copy_paste_first' \
                --acc_type 'best_acc' \
                --func_conv T \
                --hidden_channels 128 \
                --out_channels 256 \
                --bilevel T \
                --input_type STA