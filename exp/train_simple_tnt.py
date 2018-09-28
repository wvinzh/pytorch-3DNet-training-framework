from tqdm import tqdm
import torch
import torch.optim
import torchnet as tnt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchnet.engine import Engine
from torchvision import transforms
from mytransforms.target_transforms import ClassLabel, VideoID, MultiLabelTransForm
# from mytransforms.target_transforms import Compose as TargetCompose
# from mytransforms.spatial_transforms import (
#     Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
#     MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from datasets.shortvideo import ShortVideo,ShortVideo2
from utils.model import generate_model
from utils.opts import parse_opts
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import os,shutil
from models import resnext
import copy


def get_transforms(args,is_train = True):
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    args.scales=[1]
    if is_train:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.sample_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ])
    else:
        transform_train = None
    transform_val = transforms.Compose([
        transforms.RandomResizedCrop(args.sample_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    transform_target = MultiLabelTransForm(class_num=args.n_finetune_classes)
    return (transform_train, transform_val, transform_target)


def get_iterator(args, isTrain=True):
    _transforms = get_transforms(args)
    if isTrain:
        train_dataset = ShortVideo2(root_path=args.train_root_path,sample_duration=args.sample_duration,sample_step=args.sample_step,
                                   annotation_path=args.train_annotation_path, label_name_path=args.label_name_path,
                                   subset='training', spatial_transform=_transforms[0], target_transform=_transforms[2])
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=isTrain,
            num_workers=args.n_threads,
            pin_memory=True)
    else:
        val_dataset = ShortVideo2(root_path=args.val_root_path,sample_duration=args.sample_duration,sample_step=args.sample_step,
                                 annotation_path=args.val_annotation_path, label_name_path=args.label_name_path,
                                 subset='val', spatial_transform=_transforms[1], target_transform=_transforms[2])
        data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=isTrain,
            num_workers=args.n_threads,
            pin_memory=True)
    return data_loader


def save_checkpoint(state, args, is_best, epoch):
    checkpoint_filename = 'checkpoint.pth.tar'
    save_path = args.result_path
    # save_path = '.'
    checkpoint_filepath = os.path.join(save_path, checkpoint_filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(state, checkpoint_filepath)
    if is_best:
        best_filepath = os.path.join(save_path, 'best-{}.pth.tar'.format(epoch))
        shutil.copy(checkpoint_filepath,best_filepath)
# save_checkpoint({1:2},None,True,5)

def calculate_accuracy(outputs, targets, thresh_hold=0.5 ,video_id = None):
    tmp_targets = targets.type(torch.ByteTensor).cuda()
    batch_size = tmp_targets.size(0)
    pred = outputs > thresh_hold
    indexes = (pred==1)
    # if sum(indexes) <= 0:
    #     m_indexes = outputs.max(1)[0].unsqueeze(1)
    #     indexes = indexes >= m_indexes
    label_total = tmp_targets.sum(dim=1).double()
    pred_right = (pred*tmp_targets).sum(dim=1).double()
    acc = (pred_right/label_total).sum().cpu().type(torch.FloatTensor)
    # print(acc, batch_size)
    return acc, batch_size, indexes


def main():
    args = parse_opts()
    engine = Engine()
    # define meters and loggers
    loss_meter = tnt.meter.AverageValueMeter()
    map_meter = tnt.meter.mAPMeter()
    acc_meter = tnt.meter.AverageValueMeter()
    batch_meter = tnt.meter.AverageValueMeter()
    # batchtime_meter = tnt.meter.AverageValueMeter()
    env_name = 'AIChallenger-exp2'
    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Train Loss'})
    train_map_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Train mAP'})
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Test Loss'})
    test_map_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Test mAP'})
    train_acc_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Train acc'})
    test_acc_logger = VisdomPlotLogger(
        'line', port=port, env=env_name, opts={'title': 'Test acc'})

    # generate model
    model, params = generate_model(args)
    # print(model)
    best_prec1 = 0
    # ==============define loss function (criterion),optimezer,scheduler
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    
    # for param_group in optimizer.state_dict()['param_groups']:
    #             print(param_group)
        # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.begin_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # for param_group in optimizer.state_dict()['param_groups']:
            #     print(param_group)
            for param_group in optimizer.param_groups:
                # print(param_group['lr'])
                # if param_group['lr'] < 0.01:
                param_group['weight_decay'] /= 100 

            del checkpoint
            # optimizer.state_dict()['param_groups'] = tmp_param_groups
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for param_group in optimizer.state_dict()['param_groups']:
                print(param_group)

    scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)

    def h(sample, isTrain=True):
        if isTrain:
            model.train()
        else:
            model.eval()
        output = model(sample[0])
        target = sample[1]
        # print(target.size(),output.size())
        return criterion(output, target), output

    def reset_meters():
        map_meter.reset()
        loss_meter.reset()
        acc_meter.reset()
        batch_meter.reset()

    def on_sample(state):
        if not args.no_cuda:
            state['sample'] = [s.cuda() for s in state['sample'][0:2]]

    def on_forward(state):
        #using sigmoid to fit MultiLabelSoftMarginLoss
        _output = torch.sigmoid(state['output'].data)
        _target = state['sample'][1]
        acc, n_batch, _ = calculate_accuracy(_output, _target)
        map_meter.add(_output, _target)
        loss_meter.add(state['loss'].item())
        acc_meter.add(acc, n_batch)
        if(state['t'] % 100 ==0):
            print('Batch-%d loss: %.4f, accuracy: %.4f' %
              (state['t'], state['loss'].item(), acc))

    def on_start(state):
        state['best_score'] = best_prec1  # to save the best score

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('Training loss: %.4f, accuracy: %.4f,map: %.4f%%' %
              (loss_meter.value()[0], acc_meter.value()[0], map_meter.value()[0]))
        train_loss_logger.log(state['epoch'], loss_meter.value()[0])
        train_map_logger.log(state['epoch'], map_meter.value()[0])
        train_acc_logger.log(state['epoch'], acc_meter.value()[0])

        # do validation at the end of each epoch
        reset_meters()
        engine.test(h, get_iterator(args, False))
        test_loss_logger.log(state['epoch'], loss_meter.value()[0])
        test_map_logger.log(state['epoch'], map_meter.value()[0])
        test_acc_logger.log(state['epoch'], acc_meter.value()[0])
        # remember best map and save checkpoint
        now_acc = acc_meter.value()[0]
        scheduler.step(loss_meter.value()[0])
        # print(now_acc.type(),state['best_score'].type())
        is_best = now_acc > state['best_score']
        state['best_score'] = max(now_acc, state['best_score'])
        save_checkpoint({
            'epoch': state['epoch'] + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': state['best_score'],
            'optimizer': optimizer.state_dict(),
        }, args,is_best, state['epoch'])

        print('Testing loss: %.4f, accuracy: %.4f,map: %.4f%%' %
              (loss_meter.value()[0], acc_meter.value()[0], map_meter.value()[0]))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start

    engine.train(h, get_iterator(args, True),
                 maxepoch=args.n_epochs, optimizer=optimizer)

def test_on_testset():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_opts()
    data_loader = get_iterator(args,isTrain=False)
    acc_meter = tnt.meter.AverageValueMeter()
    model = resnext.resnet101(
                num_classes=args.n_finetune_classes,
                shortcut_type=args.resnet_shortcut,
                cardinality=args.resnext_cardinality,
                sample_size=args.sample_size,
                sample_duration=args.sample_duration)
    model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(torch.load('trained_models/checkpoint.pth.tar')['state_dict'])
    model.eval()
    total = 0
    result = {}
    with torch.no_grad():
        for data in data_loader:
            input = data[0].cuda()
            label = data[1].cuda()
            video_id = data[2]
            output = torch.sigmoid(model(input))
            label_indexes = (label==1)
            acc,bt,_ = calculate_accuracy(output,label ,video_id = video_id,thresh_hold=0.4)
            for i,vid in enumerate(video_id):
                if vid not in result:
                    result[vid] = []
                    result[vid].append(label_indexes[i].nonzero().squeeze(1).tolist())
                
                if sum(output[i]>0.4) > 0:
                    indexes = (output[i]>0.4)
                else:
                    indexes = (output[i]>=output[i].max(0)[0])
                tmp_index = indexes.nonzero()
                if len(tmp_index) <=0:
                    indx = []
                else:
                    indx = indexes.nonzero().squeeze(1).tolist()
                result[vid].append(indx)

            total += bt
            acc_meter.add(acc,bt)
            print('Now tested %d samples,batch Average Acc is %.4f, Average Acc is %.4f' %(total,acc/bt,acc_meter.value()[0]))
            # print(result)
    torch.save(result,'./result14-0.4-max.pkl')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
    # test_on_testset()
