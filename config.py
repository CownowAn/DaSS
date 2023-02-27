#------ PATH ------
DEFAULT_DATA_PATH='./dataset'
TEACHER_PATH='./checkpoint/teacher'

SEARCH_CANDIDATE_PATH='./preprocessed/search'
SEARCH_SAVE_PATH='./exp/search'

#------ Search Space ------
SEARCH_SPACE='resnet'

#------ Selected Nets ------
CORR_NET_PATH='./preprocessed/net_info/net_samples_50_for_correlation.pt'
NET_PATH='./preprocessed/net_info'

#------ Teacher ------
TEACHER={
    'tc_net_name': 'resnet42',
    'tc_stage_num': 4,
    'tc_stage_depth': 5, 
    'tc_stage_default_cw': [16, 32, 64, 128],
    'tc_stage_strides': [1, 2, 2, 2],
    'cw_mul': 2,
}

SPACE_CONFIG={
    'depth_list': [1, 2, 3, 4, 5],
    'cw_mult_min': 0.5,
    'cw_mult_max': 1,
    'cw_mult_step': 0.0625
}
#----- Meta-Test -----
META_TEST={
     'meta_test_datasets': ['quickdraw', 'cub', 'stanford_cars', 'dtd']
}
#----- Dataset ----
NUM_CLASSES = {
    "cub": 200,
    "dtd": 47,
    "quickdraw": 345,
    "stanford_cars": 196,
    'tiny_imagenet': 40,
}