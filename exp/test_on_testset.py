import torch
import torch.nn as nn
from utils.model import generate_model
from utils.opts import parse_opts
from datasets.shortvideo import ShortVideo
from mytransforms.target_transforms import ClassLabel, VideoID, MultiLabelTransForm
from mytransforms.target_transforms import Compose as TargetCompose
from mytransforms.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import random

def get_transforms(args):
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    args.scales=[1]
    transform_train = Compose([
        MultiScaleRandomCrop(args.scales, args.sample_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    transform_val = Compose([
        MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c']),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    transform_target = MultiLabelTransForm(class_num=args.n_finetune_classes)
    return (transform_train, transform_val, transform_target)

def main():
    #define dataset
    args = parse_opts()
    val_dataset = ShortVideo(root_path=args.val_root_path,sample_duration=args.sample_duration,sample_step=args.sample_step,
                                 annotation_path=args.val_annotation_path, label_name_path=args.label_name_path,
                                 subset='val', spatial_transform=get_transforms(args)[1], target_transform=get_transforms(args)[2])
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True)
    #define model and load parameters

def create_valset(txt_path):
    with open(txt_path,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        al = len(lines)
        val_lines = lines[0:int(al/2)]
    with open('val.txt','w') as f:
        f.writelines(val_lines)

if __name__ == '__main__':
    create_valset('/home/zengh/Dataset/AiChallengerFrames/new_test.txt')
