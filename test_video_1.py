import torch
import os
import time
from PIL import Image
from models import resnext
from utils.opts import parse_opts
import torch.nn as nn
import pims
from mytransforms.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import cv2
import pylab
import imageio
import skimage
import numpy as np
pylab.switch_backend('agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        transform.randomize_parameters()
        clip = [transform(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    return clip

# def make_data(frames_indexes,)

def test(frames,model):
    with torch.no_grad():
        output = model(frames)
        #print("output",output)
        output = torch.sigmoid(output) > 0.5
        indexs = (output==1).nonzero()
    return indexs


def predict(frames):
    start_time = time.time()
    #998875634.mp4  998104369.mp4 997020967.mp4
    cap = cv2.VideoCapture('/home/zengh/Dataset/AIChallenger/group5/997020967.mp4')
    #v = pims.Video('/home/zengh/Dataset/AIChallenger/group5/982006190.mp4')
    duration = (time.time() - start_time) * 1000
    print('cv video time %.3f ms' % duration)  

    opt = parse_opts()
    start_time = time.time()
    model = resnext.resnet101(
                num_classes=opt.n_finetune_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(torch.load('./trained_models/best.pth.tar')['state_dict'])
    duration = (time.time() - start_time)*1000
    print('restore time %.3f ms' % duration)  
  
    #model = nn.DataParallel(model)
    
    model.eval()
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    opt.scales=[1]
    transform_val = Compose([
        MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    start_time = time.time()
    clip = video_loader(root_path='/home/zengh/Dataset/AIChallenger/train/group5/567700300',frame_indices=range(3,19),transform=transform_val)
    clip = clip.unsqueeze(0)
    duration = (time.time() - start_time)*1000
    print('pic time %.3f ms' % duration)  
    #print("clip",clip.shape)
    start_time = time.time()
    indexes = test(clip, model)
    duration = (time.time() - start_time)*1000
    print('pre time %.3f ms' % duration)  

'''    
def main():
    start_time = time.time()
    #998875634.mp4  998104369.mp4 997020967.mp4
    video_path = '/home/zengh/Dataset/AIChallenger/group5/998875634.mp4'

    if os.path.exists(video_path):
        print("exists!")

    vid = imageio.get_reader(video_path, 'ffmpeg')
    

    duration = (time.time() - start_time) * 1000
    print('imageio video time %.3f ms' % duration)  '''


def main():
    
    #994513477.mp4 995153247.mp4 996259932.mp4
    start_time = time.time()
    video_path = '/home/zengh/Dataset/AIChallenger/group5/995153247.mp4'
    if os.path.exists(video_path):
        print("exists!")
    cap = cv2.VideoCapture(video_path) #15ms
    duration = (time.time() - start_time) * 1000
    print('1 time %.3f ms' % duration)  

    start_time = time.time()
    print(id(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_FRAMES,50) #40ms
    #print("id",id(cv2.CAP_PROP_POS_FRAMES))
    duration = (time.time() - start_time) * 1000
    print('2 time %.3f ms' % duration)  

    start_time = time.time()
    ret, frame = cap.read() #1ms
    duration = (time.time() - start_time) * 1000
    #print("ret",ret)
    print('3 time %.3f ms' % duration)  

    '''
    count = 1
    frames = []
    
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
        if  count % 5 == 0:
            frames.append(frame)
        count = count + 1'''

    #v = pims.Video('/home/zengh/Dataset/AIChallenger/group5/982006190.mp4')
    #duration = (time.time() - start_time) * 1000
    #print('cv video time %.3f ms' % duration)  

    opt = parse_opts()
    start_time = time.time()
    model = resnext.resnet101(
                num_classes=opt.n_finetune_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(torch.load('./trained_models/best.pth.tar')['state_dict'])
    duration = (time.time() - start_time)*1000
    print('restore time %.3f ms' % duration)  
  
    #model = nn.DataParallel(model)
    
    model.eval()
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    opt.scales=[1]
    transform_val = Compose([
        MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])
    start_time = time.time()
    clip = video_loader(root_path='/home/zengh/Dataset/AIChallenger/train/group5/567700300',frame_indices=range(3,19),transform=transform_val)
    clip = clip.unsqueeze(0)
    duration = (time.time() - start_time)*1000
    print('pic time %.3f ms' % duration)  
    #print("clip",clip.shape)
    start_time = time.time()
    indexes = test(clip, model)
    duration = (time.time() - start_time)*1000
    print('pre time %.3f ms' % duration)  
    
    
    

if __name__ == '__main__':
    main()