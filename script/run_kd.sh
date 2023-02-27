TASK=kd_student
GPU=$1
DSNAME=$2 # 'quickdraw' 'cub' 'stanford_cars' 'dtd' 

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$GPU python train_kd_net.py \
        --gpu $GPU \
        --task $TASK \
        --ds_name $DSNAME \
        --manual_seed $SEED \
        --folder_name 'dass' \
        --mode 'meta_test' \
        --pr_type 'copy_paste_first' \
        --kd_starting_lr 0.05 \
        --kd_epochs 50 \
        --alpha 0.5 \
        --temp 6. \
        --proxy_type 'dass'
done