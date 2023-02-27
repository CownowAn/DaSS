import os
import argparse
import torch
import yaml
import time
import random
import numpy as np
from collections import defaultdict
from meta_train import Meta
from sampler import TaskSampler
from myutils import str2bool, set_gpu
from config import META_TEST, SEARCH_SAVE_PATH


def main():
    args = set_args()
    device = set_gpu(args)
    
    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    file_path = os.path.join(args.save_path, 'obtained_net_index.yaml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.Loader)
    else:
        yaml_dict = {
            'net_index': defaultdict(dict),
            }

    for ds_name in META_TEST['meta_test_datasets']:
        topk_net_idx = rapid_search(args, ds_name, device)
        print(f'Search Result ==> {ds_name} | {args.proxy_type} | {topk_net_idx}\n')

        yaml_dict['net_index'][ds_name][args.proxy_type] = topk_net_idx

        os.makedirs(os.path.join(args.save_path), exist_ok=True)
        with open(os.path.join(args.save_path, f'obtained_net_index.yaml'), 'w') as f:
            yaml.dump(yaml_dict, f)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--manual_seed', type=int, default=0)
    parser.add_argument('--proxy_type', type=str, default='dass')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--load_path', type=str, default='./exp/meta_train')
    parser.add_argument('--save_path', type=str, default='./exp/search')
    parser.add_argument('--net_path', type=str, default='./preprocessed/search')
    args = parser.parse_args()
    return args


def rapid_search(args, ds_name, device):
    _args = torch.load(os.path.join(args.load_path, args.proxy_type, 'logs.pt'))['args']

    meta = Meta(_args, main_path=None, device=device, search=True)
    meta.model.load_state_dict(torch.load(os.path.join(args.load_path, args.proxy_type, 'checkpoint', 'model_best.pth.tar'))['state_dict'])
    sampler = TaskSampler(
                    args=_args,
                    mode='meta_test',
                    ds_name=ds_name,
                    ds_split=None,
                    image_size=_args.image_size,
                    batch_size=_args.batch_size,
                    n_s=_args.n_s,
                    n_q=_args.n_q,
                    bilevel=_args.bilevel,
                    tc_spp=_args.tc_spp, 
                    search=True)
    
    candidates = sampler.get_nas_task(net_path=args.net_path)
    
    start_time = time.time()
    if _args.bilevel:
        query_y_hat = meta.forward_bilevel(candidates, test=True, search=True)
    else:
        query_y_hat = meta.forward(candidates, test=True, search=True)
    query_y_hat = query_y_hat.view(-1)

    # Score
    score_list = query_y_hat.tolist()
    _, topk_net_idx = torch.topk(torch.tensor(score_list), args.topk, largest=True)
    
    return topk_net_idx.item()


if __name__ == '__main__':
    main()

