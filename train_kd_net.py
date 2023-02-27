import argparse
import os
import time
import tqdm
import argparse
import yaml
import torch
import torch.cuda
import torch.optim
import torch.utils.data

from myutils import Logger, set_seed, set_gpu, accuracy
from myutils import save_model, AverageMeter
from myutils import str2bool
from myutils import get_optimizer, get_scheduler, get_lr
from networks.resnet_nn import get_resnet
from remap import PR
from get_dataloader import get_dataloader
from loss import AlphaDistillationLoss
from config import TEACHER, TEACHER_PATH, CORR_NET_PATH, DEFAULT_DATA_PATH, NET_PATH, SEARCH_SAVE_PATH


NB_SECTIONS = 4

def set_args():
    import argparse
    from myutils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--default_data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--wandb_project_name', type=str, default="dass") 
    parser.add_argument('--manual_seed', type=int, default=0) 
    parser.add_argument('--task', type=str, default='log', help='nas_kd | kd_selected_student')
    parser.add_argument('--folder_name', type=str, default='debug')
    parser.add_argument('--mode', type=str, default='meta_train',
                        help='net_info_path is differ with the mode')    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--ds_name', type=str, default='cub')
    parser.add_argument('--ds_split', type=str, default=None)
    parser.add_argument('--valid_frequency', type=int, default=5)
    # KD Hyperparams
    parser.add_argument('--kd_optimizer_type', type=str, default='sgd')
    parser.add_argument('--kd_starting_lr', type=float, default=0.05)
    parser.add_argument('--kd_lr_schedule_type', type=str, default='cosine')
    parser.add_argument('--kd_epochs', type=int, default=50) 
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=6.)
    # Rest of the hyperparams
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=3e-5) 
    parser.add_argument('--nesterov', type=str2bool, default=True) 
    # PR
    parser.add_argument('--pr_type', type=str, default='copy_paste_first')
    # NAS
    parser.add_argument('--proxy_type', type=str, default='dass')
    parser.add_argument('--topk', type=int, default=1)
    # To collect student trained info
    parser.add_argument('--net_index', type=int, default=None)

    args = parser.parse_args()
    return args


def main():
    args = set_args()
    
    if args.ds_name != 'tiny_imagenet':
        args.ds_split = None

    if args.task == 'kd_student':
        main_path = f'./exp/{args.task}/{args.folder_name}'
        main_path += f'/{args.ds_name}'
        main_path += f'/seed-{args.manual_seed}'
        csv_path = f'./exp/{args.task}/{args.folder_name}'
        args.csv_path = csv_path
    elif args.task == 'kd_selected_student':
        main_path = f'./exp/{args.task}/{args.folder_name}'
        main_path += f'/{args.ds_name}/net-{args.net_index}'

    print(f'==> main path : {main_path}')
    os.makedirs(main_path, exist_ok=True)

    device = set_gpu(args)
    # Determinism
    set_seed(args.manual_seed)

    # Dataloader
    train_loader, valid_loader, n_classes = get_dataloader(default_data_path=args.default_data_path, 
                                                            mode=args.mode,
                                                            image_size=args.image_size,
                                                            batch_size=args.batch_size,
                                                            ds_name=args.ds_name,
                                                            ds_split=args.ds_split)
    ds = args.ds_name if args.ds_split==None else f'{args.ds_name}-{args.ds_split}'
    print(f'==> Load Data {ds}')

    # Set teacher and student 
    teacher, student, net_info = setup_teacher_student(args, n_classes, device)
    print("==> Load and Created Teacher + Student Models")

    # Criterion
    kd_loss = AlphaDistillationLoss(temperature=args.temp, alpha=args.alpha)

    total_st_time = time.time()
    
    # kd
    kd(args, teacher, student, net_info, train_loader, valid_loader, kd_loss, main_path)
    
    total_elapsed_time = time.time() - total_st_time
    print(f'Total Time: {int(total_elapsed_time//60)} (m) {int(total_elapsed_time%60)} (s)')


def kd(args, teacher, student, net_info, train_loader, valid_loader, kd_loss, main_path):
    save_path = os.path.join(main_path, 'checkpoint')
    exp_name = f'{main_path}'.replace('/', '_')
    exp_suffix = "" 
    os.makedirs(save_path, exist_ok=True)
    
    # Logs
    logger = Logger(
                log_dir=main_path,
                exp_name=exp_name,
                exp_suffix=exp_suffix,
                write_textfile=True,
                use_wandb=False,
                wandb_project_name=args.wandb_project_name,
                )
    logger.update_config(args, is_args=True)

    logger.update_config({
        'net_index': args.net_index,
        'flops': net_info[0],
        'params': net_info[1],
        'depth_config': net_info[2],
        'channel_widths': net_info[3]
    })
    
    optimizer = get_optimizer(args.kd_optimizer_type, args.kd_starting_lr, student.parameters(),
                            args.weight_decay, args.momentum, args.nesterov)
    train_steps = args.kd_epochs * len(train_loader)
    scheduler = get_scheduler(args.kd_lr_schedule_type, optimizer, train_steps)
    
    val_loss, val_acc1, val_acc5 = fine_tune_epoch(teacher, student, optimizer, scheduler, valid_loader, kd_loss, train=False)
    logger.update_config({
                        'init_valid_loss': val_loss,
                        'init_valid_top1': val_acc1,
                        'init_valid_top5': val_acc5
                        })
    
    
    ft_epoch = args.kd_epochs
    best_acc = -1
    pbar = tqdm.trange(ft_epoch)
    for epoch in pbar:
        st_epoch_time = time.time()
        train_loss, train_acc1, train_acc5 = fine_tune_epoch(teacher, student, optimizer, scheduler, train_loader, kd_loss, train=True)

        pbar.set_description(f'FT [{epoch+1}/{args.kd_epochs}] | '
                +f'TR Loss {train_loss:.2f} | TR Acc1 {train_acc1:.2f}')
        if (epoch + 1) % args.valid_frequency == 0:
            val_loss, val_acc1, val_acc5 = fine_tune_epoch(teacher, student, optimizer, scheduler, valid_loader, kd_loss, train=False)
            
            is_best = val_acc1 > best_acc
            best_acc = max(val_acc1, best_acc)

            logger.write_log_nohead({
                    'epoch': epoch+1,
                    'train/loss': train_loss,
                    'train/top1': train_acc1,
                    'train/top5': train_acc5,
                    'valid/loss': val_loss,
                    'valid/top1': val_acc1,
                    'valid/top5': val_acc5,
                    'valid/best_acc': best_acc,
                    'epoch_time': time.time() - st_epoch_time
                }, step=epoch+1)

            save_model({'epoch': epoch+1,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': student.state_dict(),
                        }, save_path, is_best=is_best, model_name=None)

            print(f'Valid Loss {val_loss:.2f} | Valid Acc1 {val_acc1:.2f}')
            print(f'Best Acc so far: {best_acc: .2f}')
    logger.save_log()
    save_csv_file(args, args.csv_path, best_acc)


def save_csv_file(args, csv_path, best_acc):
    import csv
    from os import path

    file_name = os.path.join(csv_path, f'result.csv')
    if not path.exists(file_name):
        f = open(os.path.join(csv_path, f'result.csv'), 'w', newline='')
        wr = csv.writer(f)
        row = ['data', 'proxy', 'seed', 'best acc']
        wr.writerow(row)
    else:
        f = open(os.path.join(csv_path, f'result.csv'), 'a', newline='')
        wr = csv.writer(f)
    row = [args.ds_name, args.proxy_type, args.manual_seed, round(best_acc.item(), 3)]
    wr.writerow(row)
    
    f.close()


def fine_tune_epoch(teacher, student, optimizer, scheduler, loader, kd_loss, train=True):
    teacher.eval()
    if train:
        student.train()
    else:
        student.eval()

    losses = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies5 = AverageMeter()

    for i, (inp, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        inp = inp.cuda().detach()

        with torch.no_grad():
            teacher_out = teacher(inp)

        with torch.set_grad_enabled(train):
            student_out = student(inp)
            loss = kd_loss(student_out, teacher_out, target)
            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()
                student.zero_grad()

        with torch.no_grad():
            prec1, prec5 = accuracy(student_out, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            accuracies1.update(prec1[0], inp.size(0))
            accuracies5.update(prec5[0], inp.size(0))

    return losses.avg, accuracies1.avg, accuracies5.avg


def load_teacher_net(n_classes):
    cw_mul = TEACHER['cw_mul']
    tc_stage_num = TEACHER['tc_stage_num']
    tc_stage_depth = TEACHER['tc_stage_depth']
    tc_stage_default_cw = TEACHER['tc_stage_default_cw']
    tc_stage_cws = [int(cw_mul * w) for w in tc_stage_default_cw]
    tc_stage_strides = TEACHER['tc_stage_strides']

    tc_dc = [tc_stage_depth] * tc_stage_num
    tc_cws = [[w] * tc_stage_depth for w in tc_stage_cws]
    teacher_model = get_resnet(n_classes, 
                            depth_config=tc_dc,
                            channel_widths=tc_cws, 
                            stage_strides=tc_stage_strides, 
                            tc_stage_cws=tc_stage_cws)
    return teacher_model
    

def setup_teacher_student(args, n_classes, device):
    # Teacher
    teacher_model = load_teacher_net(n_classes)
    if args.ds_name == 'tiny_imagenet':
        mode = 'meta_train'
        ds_key = f'{args.ds_name}-{args.ds_split}'
    else:
        mode = 'meta_test'
        ds_key = args.ds_name
    
    tc_net_ckpt_path = f'{TEACHER_PATH}/{ds_key}/model_best.pth.tar'
    tc_stdict = torch.load(tc_net_ckpt_path)['state_dict']

    for key in list(tc_stdict.keys()):
        if 'module.' in key:
            tc_stdict[key.replace('module.', '')] = tc_stdict.pop(key)
    teacher_model.load_state_dict(tc_stdict)

    # Student
    if args.task == 'kd_selected_student':
        net_info = torch.load(CORR_NET_PATH)[args.net_index][1]
        flops = net_info[0]
        params = net_info[1]
        st_depth_config = net_info[2]
        st_channel_widths = net_info[3]
    else:
        net_infos = torch.load(f'{NET_PATH}/net_samples.pt')
        yaml_file = f'{SEARCH_SAVE_PATH}/obtained_net_index.yaml'

        with open(yaml_file, 'r') as stream:
            parsed_yaml = yaml.load(stream, Loader=yaml.Loader)
            net_index = parsed_yaml['net_index'][args.ds_name][args.proxy_type]

        flops, params, st_depth_config, st_channel_widths = net_infos[net_index][:4]
        net_info = [flops, params, st_depth_config, st_channel_widths]

    cw_mul = TEACHER['cw_mul']
    tc_stage_default_cw = TEACHER['tc_stage_default_cw']
    tc_stage_cws = [int(cw_mul * w) for w in tc_stage_default_cw]
    student_model = get_resnet(n_classes, 
                            depth_config=st_depth_config, 
                            channel_widths=st_channel_widths, 
                            stage_strides=TEACHER['tc_stage_strides'], 
                            tc_stage_cws=tc_stage_cws)

    ## Parameter Reampping
    if args.pr_type != 'random_init':
        ## Copy and Paste Stem and Tail
        st_stdict = student_model.state_dict()
        for k, v in teacher_model.state_dict().items():
            if k == 'conv1.weight':
                sv = st_stdict[k]
                st_stdict[k] = v[:sv.size(0), :sv.size(1)]
            elif k.startswith('bn1') and k != 'bn1.num_batches_tracked':
                sv = st_stdict[k]
                st_stdict[k] = v[:sv.size(0)]
        student_model.load_state_dict(st_stdict)
        student_model.fc.load_state_dict(teacher_model.fc.state_dict())
        ## Remap Parameters stage-wisely
        param_remapper = PR(device=device, n_stage=NB_SECTIONS, tc_net=teacher_model, 
                            st_net=student_model, st_dc=st_depth_config,
                            st_cws=st_channel_widths, pr_type=args.pr_type, args=args)
        st_stages = [student_model.layer1, student_model.layer2, student_model.layer3, student_model.layer4]
        st_dict_lists = param_remapper.param_remapping()
        for i in range(NB_SECTIONS):
            print(f'=> PR. Stage-{i}')
            for d in range(st_depth_config[i]):
                st_stages[i][d].load_state_dict(st_dict_lists[i][d])                
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    print('st_depth_config')
    print(st_depth_config)
    print('st_channel_widths')
    print(st_channel_widths)

    # freeze teacher model
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    return teacher_model, student_model, net_info


if __name__ == '__main__':
    main()
