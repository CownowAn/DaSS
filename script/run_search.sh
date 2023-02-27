GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python search.py --gpu $GPU \
                                            --manual_seed 0 \
                                            --load_path './exp/meta_train' \
                                            --save_path './exp/search' \
                                            --net_path './preprocessed/search'