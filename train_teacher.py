import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.optim
from train_kd_net import load_teacher_net
from get_dataloader import get_dataloader
from myutils import AverageMeter, get_criterion, get_optimizer, get_scheduler, get_lr
from myutils import save_model
from myutils import accuracy
from myutils import Logger
from myutils import str2bool
from myutils import set_seed, set_gpu
from config import DEFAULT_DATA_PATH


parser = argparse.ArgumentParser()
#------ General ------
parser.add_argument('--default_data_path', type=str, default=DEFAULT_DATA_PATH)
parser.add_argument('--folder_name', type=str, default='dass')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--ds_name', type=str, default='tiny_imagenet')
parser.add_argument('--ds_split', type=int, default=None)
parser.add_argument('--valid_frequency', type=int, default=5) 
parser.add_argument('--manual_seed', type=int, default=0) 
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=180)
#------ Optimizer ------
parser.add_argument('--lr', type=float, default=0.05) 
parser.add_argument('--weight_decay', type=float, default=3e-5) 
parser.add_argument('--optimizer_type', type=str, default='sgd') 
parser.add_argument('--lr_schedule_type', type=str, default='cosine') 
parser.add_argument('--momentum', type=float, default=0.9) 
parser.add_argument('--nesterov', type=str2bool, default=True) 
#------ Criterion ------
parser.add_argument('--label_smoothing', type=float, default=0.1) 
parser.add_argument('--mixup_alpha', type=float, default=None)
parser.add_argument('--mul_seeds_on', type=str2bool, default=False)
#----- for meta-training with heterogenous tasks ----
parser.add_argument('--mode', type=str, default='meta_train')
parser.add_argument('--aug', type=str2bool, default=False)

args = parser.parse_args()


main_path = f'../exp/train_teacher/{args.folder_name}'
if args.ds_name == 'tiny_imagenet':
    main_path += f'/{args.ds_name}-{args.ds_split}'
else:
    main_path += f'/{args.ds_name}'
main_path += f'/lr-{args.lr}'

if args.mul_seeds_on:
    main_path += f'/seed-{args.manual_seed}'

save_path = os.path.join(main_path, 'checkpoint')
exp_name = main_path.replace('/', '_')
exp_suffix = "" 

print(main_path)

os.makedirs(main_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

device = set_gpu(args)
set_seed(args.manual_seed)


def train():
    ## Dataloader
    mode = 'meta_train' if (args.ds_name == 'tiny_imagenet') else 'meta_test'
    train_loader, valid_loader, n_classes = get_dataloader(default_data_path=args.default_data_path, 
                                                            mode=mode,
                                                            image_size=args.image_size,
                                                            batch_size=args.batch_size,
                                                            ds_name=args.ds_name,
                                                            ds_split=args.ds_split,
                                                            aug=args.aug)
    net = load_teacher_net(n_classes)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    
    ## Optimizer
    optimizer = get_optimizer(args.optimizer_type, args.lr, net.parameters(),
        args.weight_decay, args.momentum, args.nesterov)

    ## Scheduler
    train_steps = args.n_epochs * len(train_loader)
    scheduler = get_scheduler(args.lr_schedule_type, optimizer, train_steps)

    ## Criterion
    criterion = get_criterion(
        args.mixup_alpha, args.label_smoothing, is_train=True)
    test_criterion = get_criterion(
        args.mixup_alpha, args.label_smoothing, is_train=False)
    
    ## Logs
    logger = Logger(
            log_dir=main_path,
            exp_name=exp_name,
            exp_suffix=exp_suffix,
            write_textfile=True,
            use_wandb=False,
            wandb_project_name="train_teacher",
            )
    
    logger.update_config(args, is_args=True)
        
    loss, (top1, top5) = validate(net, valid_loader, test_criterion, epoch=-1, ds_name=args.ds_name)
    logger.update_config({
        'init_valid_loss': loss,
        'init_valid_top1': top1,
        'init_valid_top5': top5
    })

    best_acc = -1
    for epoch in range(args.n_epochs):
        st_epoch_time = time.time()
        train_loss, (train_top1, train_top5) = train_one_epoch(
            epoch, net, train_loader, criterion, optimizer, scheduler, ds_name=args.ds_name)

        if (epoch + 1) % args.valid_frequency == 0:
            val_loss, (val_acc, val_acc5) = validate(
                net, valid_loader, test_criterion, epoch=epoch, ds_name=args.ds_name)
            # best_acc
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)

            logger.write_log_nohead({
                'epoch': epoch+1,
                'train/loss': train_loss,
                'train/top1': train_top1,
                'train/top5': train_top5,
                'valid/loss': val_loss,
                'valid/top1': val_acc,
                'valid/top5': val_acc5,
                'valid/best_acc': best_acc,
                'epoch_time': time.time() - st_epoch_time
            }, step=epoch+1)

            save_model({
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': net.state_dict(),
                    }, save_path, is_best=is_best, model_name=None)

    logger.save_log()


def train_one_epoch(epoch, net, train_loader, criterion, optimizer, scheduler, ds_name=None):
    net.train()

    metric = {
            'data_time': AverageMeter(),
            'losses': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter()
            }

    end = time.time()
    with tqdm(total=len(train_loader),
            desc=f'{ds_name} TR Epoch #{epoch + 1}') as t:

        for i, (images, labels) in enumerate(train_loader):
            metric['data_time'].update(time.time() - end)

            images, labels = images.to(device), labels.to(device)
            # clean gradients
            net.zero_grad()
            output = net(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            metric['top1'].update(acc1[0].item(), output.size(0))
            metric['top5'].update(acc5[0].item(), output.size(0))

            loss.backward()
            optimizer.step()
            scheduler.step()

            metric['losses'].update(loss.item(), images.size(0))

            t.set_postfix({
                'top1': metric['top1'].avg, 
                'top5': metric['top5'].avg,
                'loss': metric['losses'].avg,
                'R': images.size(2),
                'lr': get_lr(optimizer),
                'data_time': metric['data_time'].avg,
            })
            t.update(1)
            end = time.time()
    return metric['losses'].avg, (metric['top1'].avg, metric['top5'].avg)


def validate(net, data_loader, test_criterion, epoch, ds_name):
    net.eval()

    metric = {
            'losses': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter()
            }

    with torch.no_grad():
        with tqdm(total=len(data_loader),
                    desc=f'{ds_name} VL Epoch #{epoch+1}') as t:
            st_time = time.time()        
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                # compute output
                output = net(images)
                loss = test_criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                metric['top1'].update(acc1[0].item(), output.size(0))
                metric['top5'].update(acc5[0].item(), output.size(0))
                metric['losses'].update(loss.item(), images.size(0))

                t.set_postfix({
                    'top1': metric['top1'].avg, 
                    'top5': metric['top5'].avg,   
                    'loss': metric['losses'].avg,
                    'img_size': images.size(2),
                })
                t.update(1)
            elapsed_time = time.time() - st_time
    return metric['losses'].avg, (metric['top1'].avg, metric['top5'].avg)


if __name__ == "__main__":
    train()