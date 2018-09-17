import torch
import os
from PIL import Image
from models import resnext
from utils.opts import parse_opts
import torch.nn as nn
from itertools import chain
from mytransforms.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def video_loader(root_path, frame_indices,label,transform=None):
    clip = []
    for i in frame_indices:
        image_path = os.path.join(root_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            clip.append(pil_loader(image_path))
        else:
            return False
    if transform is not None:
        transform.randomize_parameters()
        clip = [transform(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    clip=clip.unsqueeze(0)
    print(clip.size())
    return clip,label

# def make_data(frames_indexes,)

def test(frames,label,model):
    output = model(frames)
    output = torch.sigmoid(output) > 0.5
    indexs = (output==1).nonzero()
    return indexs,label

def read_res(res_pkl):
    res = torch.load(res_pkl)
    all_acc = 0
    for k,v in res.items():
        name = k
        label = set(v[0])
        preds=list(chain(*v[1:]))
        # print(preds)
        preds = set(preds)
        p_right = len(label & preds)
        p_all = len(label | preds)
        acc = float(p_right) / float(p_all)
        print('%s : %.6f'%(k,acc))
        all_acc += acc
    print('End :: all acc %.6f'% (all_acc/len(res)))

def change_res(res_pkl):
    res = torch.load(res_pkl)
    all_acc = 0
    v_out = []
    # new_res = {}
    for k,v in res.items():
        for i in v:
            tmp = i
            # print(i)
            if len(i) > 1:
                tmp = torch.IntTensor(i).squeeze(1).tolist()
                # print(i,tmp)
            v_out.append(tmp)
        res[k] = v_out
        v_out = []
    # print(res)
    torch.save(res,'./result-16.pkl')

def main():
    # torch.cuda.set_device(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    opt = parse_opts()
    model = resnext.resnet101(
                num_classes=opt.n_finetune_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    model.cuda()
    # print(model.cuda())
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(torch.load('./trained_models/best.pth.tar')['state_dict'])
    modle.eval()
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    opt.scales=[1]
    transform_val = Compose([
        MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    
    clip,label = video_loader('/home/zengh/Dataset/AIChallenger/test/group1/129969',frame_indices=range(1,17),label=1,transform=transform_val)
    indes = model(clip)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0])
    # clip,label = video_loader()
    # indexes = test(clip, label, model)


if __name__=='__main__':
    # change_res('result-best.pkl')
    read_res('/home/zengh/AiChallenger/Video-Classification-Pytorch-FrameWork/result-14.pkl')