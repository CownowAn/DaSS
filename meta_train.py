import time
from tqdm import tqdm
import os
import wandb
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import rankdata
from collections import OrderedDict

import torch
import torch.nn as nn

from networks.modules import gradient_update_parameters
from predictor import PredictorModel
from sampler import TaskSampler
from myutils import Logger
from myutils import save_model
from config import DEFAULT_DATA_PATH, META_TEST


def main():
    args = set_args()
    main_path, csv_path = set_path(args)
    device = set_gpu(args)
    
    meta = Meta(args, main_path, csv_path, device)
    assert args.mode == 'meta_train'
    print('start Meta-training')
    meta.meta_training()


def set_args():
    import argparse
    from myutils import str2bool
    
    parser = argparse.ArgumentParser()
    ## General / MISC
    parser.add_argument('--wandb_project_name', type=str, default='dass')
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default='meta_train')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--manual_seed', type=int, default=2)
    parser.add_argument('--folder_name', type=str, default='debug')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)    
    parser.add_argument('--user', type=str, default='sh')
    parser.add_argument('--predictor_type', type=str, default='dass', help='dass')
    
    ## Meta-training
    parser.add_argument('--mtrn_ds_split', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--mvld_ds_split', type=str, default=[8, 9])
    parser.add_argument('--meta_lr', type=float, default=5e-4)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--mvld_frequency', type=int, default=20)
    parser.add_argument('--acc_type', type=str, default='best_acc', help='final_acc | best_acc')
    
    ## Encoding scheme
    parser.add_argument('--input_type', type=str, default='STA')
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--d_inp_dim', type=int, default=512)
    parser.add_argument('--d_out_dim', type=int, default=64)
    parser.add_argument('--nz', type=int, default=56)
    parser.add_argument('--num_sample', type=int, default=5)
    parser.add_argument('--f_inp_dim', type=int, default=256)
    parser.add_argument('--f_out_dim', type=int, default=64)
    parser.add_argument('--a_inp_dim', type=int, default=181)
    parser.add_argument('--a_out_dim', type=int, default=64)
    parser.add_argument('--pr_type', type=str, default='copy_paste_first')
    
    ## Encoding scheme
    parser.add_argument('--m_inp_dim', type=int, default=128)
    
    ## Bi-level
    parser.add_argument('--bilevel', type=str2bool, default=True)
    parser.add_argument('--num_inner_updates', type=int, default=1)
    parser.add_argument('--first_order', type=str2bool, default=True)
    parser.add_argument('--step_size', type=float, default=1e-2)
    parser.add_argument('--n_s', type=int, default=1)
    parser.add_argument('--n_q', type=int, default=50)
    parser.add_argument('--task_lr', type=str2bool, default=True)
    parser.add_argument('--tc_spp', type=str2bool, default=True)
    
    ## Conv ver. FuncEncoder
    parser.add_argument('--func_conv', type=str2bool, default=True)
    parser.add_argument('--hidden_channels', type=int, default=128) 
    parser.add_argument('--out_channels', type=int, default=256) 
    
    parser.add_argument('--load_mtrn', type=str2bool, default=True)
    parser.add_argument('--load_mvld', type=str2bool, default=True)
    parser.add_argument('--load_data', type=str2bool, default=False)

    args = parser.parse_args()
    
    if args.folder_name == 'debug':
        args.use_wandb = False
        args.mtrn_ds_split = [0]
        args.mvld_ds_split = [8]
        args.meta_test_datasets = ['cub']
        args.mvld_frequency = 2

    return args


def set_path(args):
    main_path = f'./exp/meta_train/{args.folder_name}'
    csv_path = f'./exp/meta_train/{args.folder_name}'
    print(f'==> main path {main_path}')

    return main_path, csv_path


def set_gpu(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    return device


class Meta:
    def __init__(self, args, main_path, csv_path=None, device=None, search=False):
        ## General
        self.args = args
        if not search:
            self.main_path = main_path
            self.save_path = os.path.join(self.main_path, 'checkpoint')
            os.makedirs(self.main_path, exist_ok=True)
            os.makedirs(self.save_path, exist_ok=True)
            self.exp_name = self.main_path.replace('./exp/', '')
            self.csv_path = csv_path
        self.folder_name = args.folder_name
        self.exp_suffix = "" 
        self.mode = args.mode
        self.device = device
        self.default_data_path = DEFAULT_DATA_PATH
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.acc_type = args.acc_type
        assert self.acc_type in ['final_acc', 'best_acc']
        
        ## Meta-learning
        self.mtrn_ds_split = args.mtrn_ds_split
        self.mvld_ds_split = args.mvld_ds_split
        self.meta_test_datasets = META_TEST['meta_test_datasets']
        self.num_episodes = args.num_episodes
        self.meta_lr = args.meta_lr
        self.mvld_frequency = args.mvld_frequency
        
        ## Bi-level
        self.bilevel = args.bilevel
        self.num_inner_updates = args.num_inner_updates
        self.first_order = args.first_order
        self.step_size = args.step_size
        self.task_lr = args.task_lr
        self.n_s = args.n_s
        self.n_q = args.n_q
        self.tc_spp = args.tc_spp
        self.load_mtrn = args.load_mtrn
        self.load_mvld = args.load_mvld
        
        ## Predictor
        self.predictor_type = args.predictor_type
        self.model = PredictorModel(args)
        self.model = self.model.to(self.device)
        self.model_params = list(self.model.parameters())
        
        if self.task_lr:
            self.define_task_lr_params()
            self.model_params += list(self.task_lr.values())
            
        ## Criterion
        self.criterion = nn.MSELoss().to(self.device)

        if not search:
            ## Sampler
            self.get_samplers()
            
            ## Optimizer
            self.meta_optimizer = torch.optim.Adam(self.model_params, lr=self.meta_lr)
            
            ## Scheduler
            self.scheduler = None
            
            ## Logs
            self.logger = Logger(
                log_dir=self.main_path,
                exp_name=self.exp_name,
                exp_suffix=self.exp_suffix,
                write_textfile=True,
                use_wandb=args.use_wandb,
                wandb_project_name=self.args.wandb_project_name,
            )
            self.logger.update_config(self.args, is_args=True)
            
    
    def define_task_lr_params(self):
        self.task_lr = OrderedDict()
        for key, val in self.model.named_parameters():
            self.task_lr[key] = nn.Parameter(self.step_size * torch.ones_like(val))
    
    def get_samplers(self):
        self.mtrn_samplers = self.mvld_samplers = self.mtst_samplers = None
        self.mtrn_samplers_for_test = None
        
        if self.mode == 'meta_train':
            if self.load_mtrn:
                self.mtrn_samplers = []
                self.mtrn_samplers_for_test = {}
                for ds_split in self.mtrn_ds_split:
                    self.mtrn_samplers.append(
                        TaskSampler(
                            args=self.args,
                            mode='meta_train',
                            ds_name='tiny_imagenet',
                            ds_split=ds_split,
                            image_size=self.image_size,
                            batch_size=self.batch_size,
                            n_s=self.n_s,
                            n_q=self.n_q,
                            bilevel=self.bilevel,
                            tc_spp=self.tc_spp,
                        )
                    )
            if self.load_mvld:
                self.mvld_samplers = {}
                for ds_split in self.mvld_ds_split:
                    self.mvld_samplers[f'tiny_imagenet_{ds_split}'] = \
                        TaskSampler(
                            args=self.args,
                            mode='meta_valid',
                            ds_name='tiny_imagenet',
                            ds_split=ds_split,
                            image_size=self.image_size,
                            batch_size=self.batch_size,
                            n_s=self.n_s,
                            n_q=self.n_q,
                            bilevel=self.bilevel,
                            tc_spp=self.tc_spp
                            )

            self.mtst_samplers = {}
            for ds_name in self.meta_test_datasets:
                self.mtst_samplers[f'{ds_name}'] = \
                    TaskSampler(
                        args=self.args,
                        mode='meta_test',
                        ds_name=ds_name,
                        ds_split=None,
                        image_size=self.image_size,
                        batch_size=self.batch_size,
                        n_s=self.n_s,
                        n_q=self.n_q,
                        bilevel=self.bilevel,
                        tc_spp=self.tc_spp
                        )
            print('==> load samplers')
        
        elif self.mode == 'meta_valid':
            raise NotImplementedError
        
        elif self.mode == 'meta_test':
            self.mtst_samplers = {}
            for ds_name in self.meta_test_datasets:
                self.mtst_samplers[f'{ds_name}'] = \
                    TaskSampler(
                        args=self.args,
                        mode='meta_test',
                        ds_name=ds_name,
                        ds_split=None,
                        image_size=self.image_size,
                        batch_size=self.batch_size,
                        n_s=self.n_s,
                        n_q=self.n_q,
                        bilevel=self.bilevel,
                        tc_spp=self.tc_spp
                        )
    
    def forward(self, task, test=False, search=False):
        ds_info, tc_net, support, query, tcfunc_enc = task
        n = len(query['pred_inp']['arch_enc'])
        
        if test:
            with torch.no_grad():
                query_y_hat = self.model(D=ds_info,
                                         F=tc_net,
                                         A=query['arch_info'],
                                         pred_inp=query['pred_inp'],
                                         tcfunc_enc=tcfunc_enc,
                                         n=n)
        else:
            query_y_hat = self.model(D=ds_info,
                                     F=tc_net,
                                     A=query['arch_info'],
                                     pred_inp=query['pred_inp'],
                                     tcfunc_enc=tcfunc_enc,
                                     n=n)
        if search:
            return query_y_hat
        else:
            query_y = query['y'][self.acc_type].to(self.device)
            return self.criterion(query_y_hat, query_y), query_y_hat
    

    def forward_bilevel(self, task, test=False, search=False):
        # Run inner loops to get adapted parameters (theta_t`)
        ds_info, tc_net, support, query, tcfunc_enc = task
        n = len(query['pred_inp']['arch_enc'])
        n_s = self.n_s
        params = OrderedDict(self.model.meta_named_parameters())

        for n_update in range(self.num_inner_updates):
            self.model.zero_grad()
            support_y_hat = self.model(D=ds_info,
                                        F=tc_net,
                                        A=support['arch_info'],
                                        pred_inp=support['pred_inp'],
                                        tcfunc_enc=tcfunc_enc,
                                        n=n_s,
                                        params=params)
            support_y = support['y'][self.acc_type].to(self.device)
            inner_loss = self.criterion(support_y_hat, support_y)
            
            if self.task_lr is not False:
                params = gradient_update_parameters(self.model,
                                                    inner_loss,
                                                    step_size=self.task_lr,
                                                    first_order=self.first_order)
            else:
                params = gradient_update_parameters(self.model,
                                                    inner_loss,
                                                    step_size=self.step_size,
                                                    first_order=self.first_order)
        
        if test:
            with torch.no_grad():
                query_y_hat = self.model(D=ds_info,
                                        F=tc_net,
                                        A=query['arch_info'],
                                        pred_inp=query['pred_inp'],
                                        tcfunc_enc=tcfunc_enc,
                                        n=n,
                                        params=params)
        else:
            query_y_hat = self.model(D=ds_info,
                                        F=tc_net,
                                        A=query['arch_info'],
                                        pred_inp=query['pred_inp'],
                                        tcfunc_enc=tcfunc_enc,
                                        n=n,
                                        params=params)
        if search:
            return query_y_hat
        else:
            query_y  = query['y'][self.acc_type].to(self.device)
            return self.criterion(query_y_hat, query_y), query_y_hat
        
    
    def meta_training(self):
        is_best = False
        loss_keys = ['tr_loss', 'va_loss', 'te_loss']
        min_info = {k: 100000000 for k in loss_keys}
        corr_keys = ['va_spearmanr', 'va_pearsonr', 'te_spearmanr', 'te_pearsonr']
        for k in corr_keys: min_info[k] = -1
        log_keys = loss_keys + corr_keys
        
        num_meta_batch = len(self.mtrn_samplers)
        
        # init state 
        element = {}
        last_info = {}
        
        self.model.eval()
        head_list = ['va', 'te']
        for head in head_list:
            loss_dict, pearsonr_corr_dict, spearmanr_corr_dict, \
                y_all_dict, y_pred_all_dict = self.meta_valid_test(head=head, epi=0)
            # for k, v in loss_dict.items():
            #     self.logger.update(k, v)
            for k, v in pearsonr_corr_dict.items():
                self.logger.update(k, v)
            for k, v in spearmanr_corr_dict.items():
                self.logger.update(k, v)
            element.update({# f'{head}_loss': loss_dict.keys(),
                            f'{head}_pearsonr': pearsonr_corr_dict.keys(),
                            f'{head}_spearmanr': spearmanr_corr_dict.keys()})
            # last_info.update(loss_dict)
            last_info.update(pearsonr_corr_dict)
            last_info.update(spearmanr_corr_dict)
            last_info.update(y_all_dict)
            last_info.update(y_pred_all_dict)
        
        with tqdm(total=self.num_episodes, desc=f'Meta-training') as t:
            for i_epi in range(self.num_episodes):
                st_epi_time = time.time()
                self.model.train()
                self.model.zero_grad()
                meta_loss = torch.tensor(0., device=self.device)
                
                for mb in range(num_meta_batch):
                    task = self.mtrn_samplers[mb].get_random_task()
                    if self.bilevel:
                        outer_loss, _ = self.forward_bilevel(task)
                    else:
                        outer_loss, _ = self.forward(task)
                    meta_loss += outer_loss
                
                meta_loss = meta_loss / float(num_meta_batch)
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(meta_loss)
                
                self.logger.update(key='tr_time', v=(time.time()-st_epi_time))
                self.logger.update(key='tr_loss', v=meta_loss.item())

                if (i_epi + 1) % self.mvld_frequency == 0:
                    last_info = {}
                    self.logger.reset(except_keys=['tr_loss', 'tr_time'])
                    
                    ## Meta-valid
                    self.model.eval()
                    head_list = ['va', 'te']
                    for head in head_list:
                        loss_dict, pearsonr_corr_dict, spearmanr_corr_dict, \
                            y_all_dict, y_pred_all_dict = self.meta_valid_test(head=head, epi=i_epi+1)
                        # for k, v in loss_dict.items():
                        #     self.logger.update(k, v)
                        for k, v in pearsonr_corr_dict.items():
                            self.logger.update(k, v)
                        for k, v in spearmanr_corr_dict.items():
                            self.logger.update(k, v)
                        # last_info.update(loss_dict)
                        last_info.update(pearsonr_corr_dict)
                        last_info.update(spearmanr_corr_dict)
                        last_info.update(y_all_dict)
                        last_info.update(y_pred_all_dict)

                    is_best = min_info['va_spearmanr'] < last_info['va_spearmanr'] 
                    if is_best:
                        min_info = last_info
                        print('best for meta-test is updated')
                    self.model.cpu()
                    save_model({
                                'epoch': i_epi+1,
                                'optimizer': self.meta_optimizer.state_dict(),
                                'state_dict': self.model.state_dict(),
                                'last_info': last_info,
                                'min_info': min_info,
                            }, self.save_path, is_best=is_best, model_name=None)
                    self.model.to(self.device)
                    print(f'=> save model for the best state')
                    self.logger.write_log(element=element, step=i_epi+1)
                
                t.set_postfix(self.logger.avg(['tr_loss', 'te_spearmanr', 'va_spearmanr']))
                t.update(1)

        self.logger.update_config(min_info)
        self.logger.save_log()
        self._save_csv_file(min_info)

    
    def _save_csv_file(self, min_info):
        import csv
        from os import path

        file_name = os.path.join(self.csv_path, f'{self.folder_name}_spearmanr_corr.csv')
        if not path.exists(file_name):
            f = open(os.path.join(self.csv_path, f'{self.folder_name}_spearmanr_corr.csv'), 'w', newline='')
            wr = csv.writer(f)
            row = [''] + [_ for _ in min_info.keys() if 'spearman' in _]
            wr.writerow(row)
        else:
            f = open(os.path.join(self.csv_path, f'{self.folder_name}_spearmanr_corr.csv'), 'a', newline='')
            wr = csv.writer(f)
        row = [self.predictor_type] + [round(min_info[_], 3) for _ in min_info.keys() if 'spearman' in _]
        wr.writerow(row)
        
        f.close()
        
    
    def _test_task(self, sampler):
        task = sampler.get_test_task_w_all_samples()
        _, _, _, query, _ = task
        y_all = query['y'][self.acc_type]
        if self.bilevel:
            outer_loss, y_pred_all = self.forward_bilevel(task, test=True)
        else:
            outer_loss, y_pred_all = self.forward(task, test=True)
        y_all = y_all.cpu().squeeze(1).numpy()
        y_pred_all = y_pred_all.cpu().squeeze(1).numpy()
        pearsonr_corr = pearsonr(y_all, y_pred_all)[0]
        spearmanr_corr = spearmanr(y_all, y_pred_all)[0]
        return outer_loss, pearsonr_corr, spearmanr_corr, y_all, y_pred_all
    
    
    def meta_valid_test(self, head, epi):
        assert head in ['va', 'te']
        sampler_dict = self.mvld_samplers if head == 'va' else self.mtst_samplers
        loss_dict = {}
        pearsonr_corr_dict = {}
        spearmanr_corr_dict = {}
        y_all_dict = {}
        y_pred_all_dict = {}
        
        for ds_name, sampler in sampler_dict.items():
            loss, pearsonr_corr, spearmanr_corr, y_all, y_pred_all = self._test_task(sampler)
            ## Logs
            loss_dict[f'{head}_{ds_name}_loss'] = loss.item()
            pearsonr_corr_dict[f'{head}_{ds_name}_pearsonr'] = pearsonr_corr
            spearmanr_corr_dict[f'{head}_{ds_name}_spearmanr'] = spearmanr_corr
            y_all_dict[f'{head}_{ds_name}_y_all'] = y_all
            y_pred_all_dict[f'{head}_{ds_name}_y_pred_all'] = y_pred_all
        
        ## Averaging
        loss_dict[f'{head}_loss'] = sum(loss_dict.values()) / len(loss_dict)
        pearsonr_corr_dict[f'{head}_pearsonr'] = sum(pearsonr_corr_dict.values()) / len(pearsonr_corr_dict)
        spearmanr_corr_dict[f'{head}_spearmanr'] = sum(spearmanr_corr_dict.values()) / len(spearmanr_corr_dict)
        return loss_dict, pearsonr_corr_dict, spearmanr_corr_dict, y_all_dict, y_pred_all_dict


if __name__ == '__main__':
    main()
