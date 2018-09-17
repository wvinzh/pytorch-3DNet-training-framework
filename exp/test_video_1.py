import torch
import os
from PIL import Image
from models import resnext
from utils.opts import parse_opts
import torch.nn as nn

#设置只可见gpu1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def video_loader(root_path, frame_indices,transform=None):
    clip = []
    for i in frame_indices:
        image_path = os.path.join(root_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            clip.append(pil_loader(image_path))
        else:
            return False
    if transform is not None: 
        clip = [transform(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    return clip

# def make_data(frames_indexes,)

def test(frames,model):
    output = model(frames)
    output = torch.sigmoid(output) > 0.5
    indexs = (output==1).nonzero()
    return indexs[0]

'''
def predict(frames):
    opt = parse_opts()
    model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    model.load_state_dict(torch.load('./trained_models/best.pth.tar'))
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[1])
    indexes = test(frames, model)'''

def main():
    opt = parse_opts()
    model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    model.load_state_dict(torch.load('./trained_models/best.pth.tar'))
    model = model.cuda()

    model = nn.DataParallel(model, device_ids=[0])
    clip = video_loader(root_path='/home/zengh/Dataset/AIChallenger/test/group0/1000007124',frame_indices=range(16))
    indexes = test(clip, model)
    print(indexes)