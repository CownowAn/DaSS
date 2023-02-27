GPU=$1
DSNAME=$2
FOLDER='for_correlation'

echo $DSNAME
for INDEX in {0..49}
do
    CUDA_VISIBLE_DEVICES=$GPU python train_kd_net.py --gpu $GPU \
        --manual_seed 0 \
        --task 'kd_selected_student' \
        --mode 'meta_test' \
        --kd_starting_lr 0.05 \
        --kd_epochs 50 \
        --alpha 0.5 \
        --temp 6. \
        --folder_name $FOLDER \
        --ds_name $DSNAME \
        --net_index $INDEX
done
