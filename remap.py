from collections import defaultdict
# from pr_baselines import remap_copy_paste, random_init, uniform, l1norm_sorting
from collections import defaultdict
from sklearn.decomposition import PCA
import torch
import torch.nn as nn


def _remap_copy_paste_first(tc_stage_params, st_stage_params):

    st_params = defaultdict(list)
    for i, v in enumerate(st_stage_params['conv']):
        out_ch, in_ch, k, k = v.size()
        st_params['conv'].append(tc_stage_params['conv'][i][:out_ch, :in_ch])
    
    for key in ['weight', 'bias', 'running_mean', 'running_var']:
        for i, v in enumerate(st_stage_params[key]):
            st_params[key].append(tc_stage_params[key][i][:len(st_stage_params[key][i])])

    return st_params


def _remap_copy_paste_last(tc_stage_params, st_stage_params):
    
    offset = len(tc_stage_params['conv']) - len(st_stage_params['conv'])

    st_params = defaultdict(list)
    for i, v in enumerate(st_stage_params['conv']):
        out_ch, in_ch, k, k = v.size()
        st_params['conv'].append(tc_stage_params['conv'][offset+i][-out_ch:, -in_ch:])
    
    for key in ['weight', 'bias', 'running_mean', 'running_var']:
        for i, v in enumerate(st_stage_params[key]):
            st_params[key].append(tc_stage_params[key][offset+i][-len(st_stage_params[key][i]):])

    return st_params


def _remap_copy_paste_uniform(tc_stage_params, st_stage_params, btype):
    offset = 0
    if 'last' in btype:
        offset = len(tc_stage_params['conv']) - len(st_stage_params['conv'])
    
    st_params = defaultdict(list)
    for i, v in enumerate(st_stage_params['conv']):
        out_ch, in_ch, k, k = v.size()
        t_out_ch, t_in_ch, _, _ = tc_stage_params['conv'][offset+i].size()
        out_s = int(t_out_ch / out_ch)
        in_s = int(t_in_ch / in_ch)
        params = tc_stage_params['conv'][offset+i][::out_s]
        if params.size(0) > out_ch:
            params = params[:out_ch]
        params = params[:, ::in_s]
        if params.size(1) > in_ch:
            params = params[:, :in_ch]
        st_params['conv'].append(params)
        # st_params['conv'].append(tc_stage_params['conv'][offset+i][::out_s, ::in_s])
    
    for key in ['weight', 'bias', 'running_mean', 'running_var']:
        for i, v in enumerate(st_stage_params[key]):
            s = int(len(tc_stage_params[key][offset+i]) / len(st_stage_params[key][i]))
            params = tc_stage_params[key][offset+i][::s]
            if params.size(0) > len(st_stage_params[key][i]):
                params = params[:len(st_stage_params[key][i])]
            st_params[key].append(params)
            # st_params[key].append(tc_stage_params[key][offset+i][::s])

    return st_params


def remap_copy_paste(tc_stage_params, st_stage_params, depth, btype):

    if btype == 'copy_paste_first':
        st_params = _remap_copy_paste_first(tc_stage_params, st_stage_params)
    elif btype == 'copy_paste_last':
        st_params = _remap_copy_paste_last(tc_stage_params, st_stage_params)
    elif 'copy_paste_uniform' in btype:
        st_params = _remap_copy_paste_uniform(tc_stage_params, st_stage_params, btype)
    else:
        raise ValueError(btype)
    ## Build student state_dict
    st_dict_list = []
    num_layer = 2
    for d in range(depth):
        st_dict = {}
        for j in range(num_layer):
            st_dict[f'conv{j+1}.weight'] = st_params['conv'][d*num_layer+j]
            for key in ['weight', 'bias', 'running_mean', 'running_var']:
                st_dict[f'bn{j+1}.{key}'] = st_params[key][d*num_layer+j]
            st_dict[f'bn{j+1}.num_batches_tracked'] = tc_stage_params['num_batches_tracked'][0] 
        st_dict_list.append(st_dict)
    
    return st_dict_list



class PR():
    '''
    Parameter remapping stage-wisely for whole st network
    '''
    def __init__(self, device, n_stage, tc_net, st_net, st_dc, st_cws, pr_type, args):
        ## General
        self.args = args
        self.device = device
        self.image_size = args.image_size
        
        ## PR type
        self.pr_type = pr_type

        ## Search Space setting
        self.n_stage = n_stage
        
        ## Teacher Net
        self.tc_net = tc_net
        self.tc_net.eval()
        self.tc_stages = [self.tc_net.layer1, self.tc_net.layer2, self.tc_net.layer3, self.tc_net.layer4]
        self.tc_stages = [_.to(device) for _ in self.tc_stages]
        self.tc_stage_params = [self._get_stage_params(stage) for stage in self.tc_stages]

        ## Student Net
        self.st_net = st_net
        self.st_stages = [self.st_net.layer1, self.st_net.layer2, self.st_net.layer3, self.st_net.layer4]
        self.st_stages = [_.to(device) for _ in self.st_stages]
        self.st_stage_params = [self._get_stage_params(stage) for stage in self.st_stages]
        self.st_dc = st_dc
        self.st_cws = st_cws


    def _get_stage_params(self, stage):
        stage_params = defaultdict(list)
        for block in stage:
            for k, v in block.state_dict().items():
                if 'conv' in k:
                    stage_params['conv'].append(v)
                elif 'bn' in k:
                    # weight, bias, running_mean, running_var
                    stage_params[k[4:]].append(v)
                else: raise ValueError(k)        
        return stage_params
        

    def param_remapping(self):
        st_dict_lists = []
        for i in range(self.n_stage):
            tc_stage_params = self.tc_stage_params[i]
            st_stage_params = self.st_stage_params[i]
            depth = self.st_dc[i]

            if self.pr_type is None: # No PR
                ## Build student state_dict
                st_dict_list = []
                num_layer = 2
                for d in range(self.st_dc[i]):
                    st_dict = {}
                    for j in range(num_layer):
                        st_dict[f'conv{j+1}.weight'] = self.st_stage_params[i]['conv'][d*num_layer+j]
                        for key in ['weight', 'bias', 'running_mean', 'running_var']:
                            st_dict[f'bn{j+1}.{key}'] = self.st_stage_params[i][key][d*num_layer+j]
                        st_dict[f'bn{j+1}.num_batches_tracked'] = self.tc_stage_params[i]['num_batches_tracked'][0] 
                    st_dict_list.append(st_dict)
            elif self.pr_type == 'random_init':
                random_type = 'kaiming_normal'
                st_dict_list = random_init(tc_stage_params, st_stage_params, depth, random_type)
            elif 'copy_paste' in self.pr_type:
                st_dict_list = remap_copy_paste(tc_stage_params, st_stage_params, depth, btype=self.pr_type)
                        
            else: raise ValueError(self.pr_type)
            
            st_dict_lists.append(st_dict_list)
        return st_dict_lists
