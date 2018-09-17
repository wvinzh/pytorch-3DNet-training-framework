""" Run example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!

    Example:
        $ python -m visdom.server -port 8097 &
        $ python train_with_tnt.py
"""
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.models as MODEL
from utils.engine import MulticlassEngine, MultiLabelMAPEngine
from mytransforms.temporal_transforms import LoopPadding, TemporalRandomCrop
from mytransforms.target_transforms import ClassLabel, VideoID, MultiLabel
from mytransforms.target_transforms import Compose as TargetCompose
from mytransforms.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from torchnet.logger import MeterLogger
from datasets.shortvideo import ShortVideo
from models.resnet import resnet50

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('-rp', '--root_path', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('-ap', '--annotation_path', metavar='DIR',
                    help='path to annotation_path (e.g. ../data/')
parser.add_argument('-lnp', '--label_name_path', metavar='DIR',
                    help='path to label_name (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=256, type=int,
                    metavar='N', help='image size (default: 256)')
parser.add_argument('--crop-size', '-cs', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-cn', '--class-num', default=63, type=int,
                    metavar='N', help='class num (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument(
    '--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument(
    '--lr_patience',
    default=10,
    type=int,
    help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
)
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=1, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--logger-title', default='Hao-Sensitive', type=str,
                    help='The log title to of tnt')


def main_sensitive():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    transform_train = Compose([
        Scale(args.image_size),
        CenterCrop(args.crop_size),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    transform_val = Compose([
        Scale(args.image_size),
        CenterCrop(args.crop_size),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])

    num_classes = args.class_num
    transform_target = MultiLabelTransForm(class_num=num_classes)
    # define dataset
    train_dataset = ShortVideo(root_path=args.root_path, annotation_path=args.annotation_path, label_name_path=args.label_name_path,
                               subset='training', spatial_transform=transform_train, target_transform=transform_target)
    val_dataset = ShortVideo(root_path=args.root_path, annotation_path=args.annotation_path, label_name_path=args.label_name_path,
                             subset='val', spatial_transform=transform_train, target_transform=transform_target)

    # load model
    # ==============for resnet
    model = resnet50(sample_size=112, sample_duration=16,
                     num_classes=num_classes)

    # ==============define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # ==============define optimizer,set different lr between fc and base params
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 'min', patience=args.lr_patience)
    # scheduler = lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[20, 30], gamma=0.1, last_epoch=-1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # [3,13,15,18] max 20,[5, 15, 18] max [15,30]
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'crop_size': args.crop_size, 'max_epochs': 100,
             'numclass': num_classes, 'evaluate': args.evaluate, 'resume': args.resume, 'print_freq': args.print_freq, 'logger_title': args.logger_title}

    state['save_model_path'] = './trained_models/ResNet50/'

    engine = MultiLabelMAPEngine(state)

    engine.learning(model, criterion, train_dataset,
                    val_dataset, optimizer, scheduler)


if __name__ == '__main__':
    main_sensitive()
