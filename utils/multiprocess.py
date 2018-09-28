import cv2
import threading
import queue
import random
import time
import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method('spawn')
import torch
from PIL import Image
from torchvision import transforms
from models import resnext
from opts import parse_opts
import os
import torch.nn as nn


def get_model():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    model.load_state_dict(torch.load(
        './trained_models/best.pth10.tar')['state_dict'])
    model.cpu()
    # model.cuda()
    model.eval()
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    opt.scales = [1]
    return model


def frame_writer(cap, frame_queue, all_index):
    st = time.time()
    frame_index = 0
    while True:
        # st= time.time()
        ret, frame = cap.read()
        # print('read one',time.time()-st)
        if not ret:
            frame_queue.put((-1, -1))
            break
        if frame_index in all_index:
            # frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            frame_queue.put((frame_index, frame))
        # print('%s write %d====='%(threading.current_thread().name,frame_index))
        frame_index += 1
    print('out writer,', time.time()-st)


def get_transform(is_train=False):
    if is_train:
        pass
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


def frame_reader(frame_queue, indexes, input_queue, clip_len=16):
    print('%s start=====' % threading.current_thread().name)
    r_st = time.time()
    readed_num = 0
    clip = []
    output = []
    transform = get_transform()
    while True:
        frame_index, frame = frame_queue.get()
        if frame_index == -1:
            frame_queue.put((frame_index, frame))
            break
        # if frame_index in indexes:
        readed_num += 1

        # frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        # # print(type(pil_img))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st = time.time()

        a = transform(frame)
        # print('trans',time.time()-st)
        #
        clip.append(a)
        if len(clip) == 16:
            output.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
    #         # print(output.size())
            clip.clear()
    output = torch.stack(output, 0)
    print(output.size())
    input_queue.put((1, output))
    input_queue.put((-1, -1))
    print('out reader time,', time.time()-r_st)
    # print('%s, read a frame from %d' %
    #       (threading.current_thread().name, frame_index))
    # print(threading.current_thread().name,'read %d' % readed_num)


def compute_indexes(num_of_frames, num_of_clip=4, len_of_clip=16, nousing_frames=10):
    total = num_of_frames - nousing_frames
    sampled_frame = sorted(random.sample(
        range(total), len_of_clip*num_of_clip))
    index_list = [[]]*num_of_clip
    for i in range(num_of_clip):
        index_list[i] = sampled_frame[i*len_of_clip:(i+1)*len_of_clip]
    return sampled_frame, index_list


def main(file, model):
    st = time.time()
    model.cuda()
    print('cuda time', time.time()-st)
    model.eval()
    # video_path = '/home/zengh/Dataset/AIChallenger/group5/777770896.mp4'
    cap = cv2.VideoCapture(file)
    # writer_thread.join()
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('there are %d frames' % total_frames)
    all_index, index_list = compute_indexes(int(total_frames))
    frame_queue = queue.Queue()
    input_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=frame_writer, args=(cap, frame_queue, all_index))
    writer_thread.start()

    reader_thread = threading.Thread(
        target=frame_reader, args=(frame_queue, all_index, input_queue,))
    reader_thread.start()
    reader_thread.join()
    writer_thread.join()
    while True:
        ret, input = input_queue.get()
        print('hello', ret)
        if ret == -1:
            break
        st = time.time()
        output = model(input.cuda())
        print('predict tiem', time.time()-st)

    print('All end')


def main_single(file):
    total = 0
    cap = cv2.VideoCapture(file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('there are %d frames' % total_frames)
    all_index, index_list = compute_indexes(int(total_frames))
    frames = []
    tr = get_transform()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total in all_index:
            tr(frame)
            frames.append(frame)
        total += 1
        if len(frames) == 16:
            # print()pass
            frames.clear()

    print('readed total %d' % total)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)
    # DEVICE = torch.device("cuda")

    st = time.time()
    # main()
    # model = get_model()
    opt = parse_opts()
    # torch.manual_seed(25)
    model = resnext.resnet101(
        num_classes=opt.n_finetune_classes,
        shortcut_type=opt.resnet_shortcut,
        cardinality=opt.resnext_cardinality,
        sample_size=opt.sample_size,
        sample_duration=opt.sample_duration)
    print('model load time', time.time()-st)
    # torch.manual_seed(23)
    # model.to(DEVICE)
    # # model.cpu()
    # model.eval()
    # # print(model)
    # model.share_memory()
    # print(model)
    file_list = ['/home/zengh/Dataset/AIChallenger/group5/890138181.mp4',
                 '/home/zengh/Dataset/AIChallenger/group5/914495638.mp4',
     '/home/zengh/Dataset/AIChallenger/group5/777770896.mp4', '/home/zengh/Dataset/AIChallenger/group5/904105399.mp4',
     '/home/zengh/Dataset/AIChallenger/group5/919954315.mp4', '/home/zengh/Dataset/AIChallenger/group5/919960825.mp4']
    #  '/home/zengh/Dataset/AIChallenger/group5/894736565.mp4','/home/zengh/Dataset/AIChallenger/group5/913827092.mp4']
    prs = []
    for f in file_list:
        pr = multiprocessing.Process(target=main, args=(f, model,))
        pr.start()
        prs.append(pr)
    for p in prs:
        p.join()
    print(time.time()-st)
