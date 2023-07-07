import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from models import ResNet18, ResNet50, WRN28_10
from utils.dataset import get_cifar_ddp
from utils.metrics import accuracy
from utils.logger import CSVLogger, AverageMeter
from utils.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar100", help='Dataset')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--epochs', type=int, default=200, help='Epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--mo', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--loadckpt', default=False, action='store_true')
#parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for ddp training")
args = parser.parse_args()

# set random seed
set_seed(args.seed)

# Intialize directory to save results
args.ckpt_dir = "./results"
os.makedirs(args.ckpt_dir, exist_ok=True)
logger_name = os.path.join(args.ckpt_dir, f"dp_{args.model}_{args.dataset}")

# Logging tools
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(logger_name + ".log"),
        logging.StreamHandler(),
    ],
)
logging.info(args)

def run_one_epoch(phase, loader, model, criterion, optimizer, args):
    loss, acc = AverageMeter(), AverageMeter()
    for batch_idx, inp_data in enumerate(loader, 1):
        inputs, targets = inp_data
        inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            model.train()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
        elif phase == 'val':
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logging.info('Define correct phase')
            quit()

        batch_acc = accuracy(outputs, targets, topk=(1,))[0]
        loss.update(batch_loss.item(), inputs.size(0))
        acc.update(float(batch_acc), inputs.size(0))
    return loss.avg, acc.avg

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    trainloader, testloader = get_cifar_ddp(args)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.milestones = [100, 120]
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.milestones = [100, 150]
    else:
        print(f"Unsupported dataset: {args.dataset}")

    if args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes)
    elif args.model == 'wrn':
        model = WRN28_10(num_classes=args.num_classes)
    else:
        print("Unsupported model")
        quit()

    model = model.cuda(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    
    csv_logger = CSVLogger(args, ['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'], logger_name + '.csv')

    if args.loadckpt:
        state = torch.load(f"{logger_name}_best.pth.tar")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_acc = state['best_acc']
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 0
        best_acc = -float('inf')

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        trainloader.sampler.set_epoch(epoch)
        trainloss, trainacc = run_one_epoch('train', trainloader, model, criterion, optimizer, args)
        valloss, valacc = run_one_epoch('val', testloader, model, criterion, optimizer, args)
        scheduler.step()

        if local_rank == 0:
            logging.info('Epoch: [%d | %d]' % (epoch, args.epochs))
            logging.info(f"Epoch {epoch} takes {(time.time() - t):.2f} seconds")
            logging.info('Train_Loss = {0}, Train_acc = {1}'.format(trainloss, trainacc))
            logging.info('Val_Loss = {0}, Val_acc = {1}'.format(valloss, valacc))
            csv_logger.save_values(epoch, trainloss, trainacc, valloss, valacc)

            if valacc > best_acc:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(state, f"{logger_name}_best.pth.tar")
                best_acc = valacc

            logging.info(f'best acc:{best_acc}')

if __name__ == '__main__':
    main(args)
