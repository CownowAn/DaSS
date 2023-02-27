from collections import defaultdict
import torch

from networks.resnet import get_resnet
from myutils import InfIterator
from get_dataloader import get_dataloader
from config import TEACHER, NUM_CLASSES, DEFAULT_DATA_PATH, TEACHER_PATH


class TaskSampler():
    def __init__(self, args, mode, ds_name, ds_split, image_size, 
            batch_size, n_s, n_q, bilevel, tc_spp, search=False):
        ## General
        self.args = args
        self.user = args.user
        self.search = search
        self.predictor_type = args.predictor_type
        self.mode = mode
        self.default_data_path = DEFAULT_DATA_PATH
        self.ds_name = ds_name
        self.ds_split = ds_split
        if self.ds_name == 'tiny_imagenet':
            self.ds_key = f'{self.ds_name}-{self.ds_split}'
        else:
            self.ds_key = f'{self.ds_name}'
        self.image_size = image_size
        self.batch_size = batch_size
        self.tc_net_name = TEACHER['tc_net_name'] # resnet42
        
        ## Bi-level
        self.bilevel = bilevel
        self.n_s = n_s
        self.n_q = n_q
        self.tc_spp = tc_spp

        ## Dataloader
        self.n_classes = NUM_CLASSES[self.ds_name]
        self.load_data = args.load_data
        if self.load_data:
            self.train_loader, self.valid_loader, self.n_classes = get_dataloader(
                self.mode, self.default_data_path, self.image_size, 
                    self.batch_size, self.ds_name, self.ds_split)
            self.train_img_data_iter = InfIterator(self.train_loader)
            self.valid_img_data_iter = InfIterator(self.valid_loader)
        
        ## Load teacher network
        self._load_tc_net()
        print(f'==> load {self.ds_key} task sampler')

        if not self.search:
            self._get_stnet_info()

        if self.tc_spp:
            self._get_tcnet_info()
        else:
            self._get_spp_stnet_info()

        
    def _load_tc_net(self):
        cw_mul = TEACHER['cw_mul']
        tc_stage_num = TEACHER['tc_stage_num']
        tc_stage_depth = TEACHER['tc_stage_depth']
        tc_stage_default_cw = TEACHER['tc_stage_default_cw']
        tc_stage_cws = [int(cw_mul * w) for w in tc_stage_default_cw]
        tc_stage_strides = TEACHER['tc_stage_strides']

        tc_dc = [tc_stage_depth] * tc_stage_num
        tc_cws = [[w] * tc_stage_depth for w in tc_stage_cws]
        self.tc_net = get_resnet(self.n_classes, 
                                depth_config=tc_dc,
                                channel_widths=tc_cws, 
                                stage_strides=tc_stage_strides, 
                                tc_stage_cws=tc_stage_cws)

        tc_net_ckpt_path = f'{TEACHER_PATH}/{self.ds_key}/model_best.pth.tar'
        self.tc_net.load_state_dict(torch.load(tc_net_ckpt_path)['state_dict'])        

    def _get_tcnet_info(self):
        self.tcnet_info = torch.load(f'./preprocessed/ours/{self.mode}/teacher/{self.ds_key}-logs_noise_actmap.pt')
        self.tcfunc = self.tcnet_info[f'noise_actmap_stage-4'][0]

    def _get_stnet_info(self):
        self.stnet_info = {}
        self.stnet_info = torch.load(f'./preprocessed/ours/{self.mode}/{self.ds_key}-logs_noise_actmap.pt')
        self.num_stnet = len(self.stnet_info['net_index'])

    def _get_spp_stnet_info(self):
        raise NotImplementedError
    
    def _get_spp_set(self):
        assert self.bilevel
        if self.tc_spp:
            spp_info  = self.tcnet_info
        else:
            spp_info = self.spp_stnet_info
                
        support = {'arch_info': {
                                'depth_config': [],
                                'channel_widths': [],
                                },
                    'y': {
                            'final_acc': [],
                            'best_acc': [],
                            'final_loss': [],
                            },
                    'pred_inp': {
                            'arch_enc': [],
                            'func_enc': [],
                    }}
        # arch info
        support['arch_info']['depth_config'] = spp_info['depth_config']
        support['arch_info']['channel_widths'] = spp_info['channel_widths']
        
        # y info
        final_acc_s = [torch.tensor(_) for _ in spp_info['final_acc']]
        best_acc_s =  [torch.tensor(_) for _ in spp_info['best_acc']]
        final_loss_s = [torch.tensor(_) for _ in spp_info['final_loss']]
        support['y']['final_acc'] = torch.stack(final_acc_s[:self.n_s]).view(-1, 1)
        support['y']['best_acc'] = torch.stack(best_acc_s[:self.n_s]).view(-1, 1)
        support['y']['final_loss'] = torch.stack(final_loss_s[:self.n_s]).view(-1, 1)
        # pred_inp info
        support['pred_inp']['arch_enc'] = spp_info['arch_enc']
        support['pred_inp']['func_enc'] = spp_info[f'noise_actmap_stage-4']
        
        return support


    def get_random_task(self):
        if self.load_data:
            x, _ = next(self.train_img_data_iter)
        else:
            x = None
        ds_info = {
            'ds_name': self.ds_name,
            'ds_split': self.ds_split,
            'ds_imgs': x
        }
        
        index_list = torch.randperm(self.num_stnet)[:self.n_s + self.n_q]
        st = self.stnet_info
        ## y info.
        final_acc = [torch.tensor(st['final_acc'][i]) for i in index_list]
        best_acc = [torch.tensor(st['best_acc'][i]) for i in index_list]
        final_loss = [torch.tensor(st['final_loss'][i]) for i in index_list]
            
        ## pred_inp info.
        arch_enc = [st['arch_enc'][i] for i in index_list]
        func_enc = [st[f'noise_actmap_stage-4'][i] for i in index_list]
        
        ## arch info.
        depth_config = [st['depth_config'][i] for i in index_list]
        channel_widths = [st['channel_widths'][i] for i in index_list]

        if self.bilevel:
            support = self._get_spp_set()
        else:
            support = None
        query = {'arch_info': {
                                'depth_config': depth_config[self.n_s:],
                                'channel_widths': channel_widths[self.n_s:],
                                },
                    'y': {
                            'final_acc': torch.stack(final_acc[self.n_s:]).view(-1, 1),
                            'best_acc': torch.stack(best_acc[self.n_s:]).view(-1, 1),
                            'final_loss': torch.stack(final_loss[self.n_s:]).view(-1, 1),
                            },
                    'pred_inp': {
                            'arch_enc': arch_enc[self.n_s:],
                            'func_enc': func_enc[self.n_s:],
                            # 'func_enc_tans': func_enc_tans[self.n_s:]
                    }}
        
        return ds_info, self.tc_net, support, query, self.tcfunc


    def get_test_task_w_all_samples(self):
        if self.load_data:
            x, _ = next(self.train_img_data_iter)
        else:
            x = None
        ds_info = {
            'ds_name': self.ds_name,
            'ds_split': self.ds_split,
            'ds_imgs': x
        }
        if self.bilevel:
            support = self._get_spp_set()
        else:
            support = None
        query = {'arch_info': {
                                'depth_config': [],
                                'channel_widths': [],
                                },
                    'y': {
                            'final_acc':[],
                            'best_acc': [],
                            'final_loss': [],
                            },
                    'pred_inp': {
                            'arch_enc': [],
                            'func_enc': [],
                    }}

        ## Query set
        # arch info
        idx_list = list(range(self.num_stnet))
        for idx in idx_list: # 50
            query['pred_inp']['arch_enc'].append(self.stnet_info['arch_enc'][idx])
            query['pred_inp']['func_enc'].append(self.stnet_info[f'noise_actmap_stage-4'][idx])
            query['arch_info']['depth_config'].append(self.stnet_info['depth_config'][idx])
            query['arch_info']['channel_widths'].append(self.stnet_info['channel_widths'][idx])
        # y info
        final_acc_q = [torch.tensor(_) for _ in self.stnet_info['final_acc']]
        best_acc_q =  [torch.tensor(_) for _ in self.stnet_info['best_acc']]
        final_loss_q = [torch.tensor(_) for _ in self.stnet_info['final_loss']]
        
        for idx in idx_list:
            query['y']['final_acc'].append(final_acc_q[idx])
            query['y']['best_acc'].append(best_acc_q[idx])
            query['y']['final_loss'].append(final_loss_q[idx])

        query['y']['final_acc'] = torch.stack(query['y']['final_acc']).view(-1, 1)
        query['y']['best_acc'] = torch.stack(query['y']['best_acc']).view(-1, 1)
        query['y']['final_loss'] = torch.stack(query['y']['final_loss']).view(-1, 1)

        return ds_info, self.tc_net, support, query, self.tcfunc

    
    def get_nas_task(self, net_path=None): 
        ## ds info
        if self.load_data:
            x, _ = next(self.train_img_data_iter)
        else:
            x = None
        ds_info = {
            'ds_name': self.ds_name,
            'ds_split': self.ds_split,
            'ds_imgs': x
        }
        
        ## Support set
        if self.bilevel:
            support = self._get_spp_set()
        else:
            support = None

        ## Query set
        logs_path = f'{net_path}/{self.ds_key}.pt'
        query = torch.load(logs_path)
        query['pred_inp']['func_enc'] = query[f'noise_actmap_stage-4']
        
        return ds_info, self.tc_net, support, query, self.tcfunc
        

if __name__ == '__main__':
    pass