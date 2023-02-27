GPU=$1
DATA=$2

for LR in 0.01 
do
    CUDA_VISIBLE_DEVICES=$GPU python train_teacher.py --gpu $GPU \
                            --mode 'meta_test' \
                            --folder_name 'teacher' \
                            --ds_name $DATA \
                            --lr $LR \
                            --manual_seed 0 \
done
