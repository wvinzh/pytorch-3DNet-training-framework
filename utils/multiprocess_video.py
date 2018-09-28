import threading
import multiprocessing
from multiprocessing import Queue
import os
# import queue
import threadpool
import cv2
from PIL import Image
import random
import time
from torchvision import transforms
import torch

def get_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.ToTensor(),
    ])

def transform_frame(transform,frame_queue,readed_dict,sample_queue,lock):
    while True:
        try:
            video_path,index,frame = frame_queue.get(timeout=1)
            frame = transform(frame)
        except Exception as e:
            print(e)
            break
        lock.acquire()
        if video_path in readed_dict:
            # print(type(frame),frame.size())
            readed_dict[video_path].append(frame)
            if index%16 == 15:
                # sample = torch.stack(readed_dict[video_path],0).permute(1,0,2,3)
                readed_dict[video_path].clear()
                if index // 15 == 4:
                    del readed_dict[video_path]
                    print('all frames ok')
        else:
            readed_dict[video_path] = []
            readed_dict[video_path].append(frame)
        lock.release()


def compute_indexes(num_of_frames, num_of_clip=4, len_of_clip=16, nousing_frames=10):
    total = num_of_frames - nousing_frames
    sampled_frame = sorted(random.sample(
        range(total), len_of_clip*num_of_clip))
    index_list = [[]]*num_of_clip
    for i in range(num_of_clip):
        index_list[i] = sampled_frame[i*len_of_clip:(i+1)*len_of_clip]
    return sampled_frame, index_list


def read_line(video_txt, line_queue):
    file = open(video_txt, 'r')
    line = file.readline()
    line_num = 0
    while line:
        line = line.strip()
        line_queue.put((line_num, line))
        line_num += 1
        line = file.readline()
    print('read line thread end')
    line_queue.put((-1, -1))
    file.flush()
    file.close()


def extract_video_frame(video_path, frame_queue):
    cap = cv2.VideoCapture(video_path)
    # print('extract  in')
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    all_index, index_list = compute_indexes(int(total_frames))
    frame_index = 0
    frame_sequence = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index in all_index:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_queue.put((video_path, frame_sequence, frame))
            frame_sequence += 1
        frame_index += 1


def video_reader(line_queue, frame_queue):
    while True:
        index, video_path = line_queue.get()
        if index < 0:
            line_queue.put((index, video_path))
            break
        print(video_path)
        extract_video_frame(video_path, frame_queue)
        # line_queue.task_done()
    print('video.reader end')



def test_video(video_txt):
    '''input a video txt with label
    output a txt with pridected label

    Arguments:
        root_path {string} -- where the video saved
        video_txt {string} -- the txt file saved video info
    '''
    assert os.path.exists(video_txt), 'video txt need to be a file'

    # start a line to read file and put the line in a queue
    line_queue = Queue()  # TO save line
    read_line_thread = threading.Thread(
        target=read_line, args=(video_txt, line_queue,))
    read_line_thread.start()

    # start some video reader threads  to read videos
    # and save the video frame to a frame queue
    max_video_reader_thread = 4
    v_reader_threads = []
    frame_queue = Queue()  # use the queue to save frames
    for i in range(max_video_reader_thread):
        vt = threading.Thread(target=video_reader,
                              args=(line_queue, frame_queue))
        vt.start()
        v_reader_threads.append(vt)

    # start a process to read frame and transform it
    readed = {}  # used to record video has been readed
    frame_transform = get_transform()
    lock = threading.Lock()
    tt=[]
    for i in range(2):
        t = threading.Thread(target=transform_frame,args=(frame_transform,frame_queue,readed,None,lock))
        t.start()
        tt.append(t)
    for i in tt:
        i.join()
    # while True:
    #     try:
    #         video_path,index,frame = frame_queue.get(timeout=1)
    #     except Exception as em:
    #         print(em)
    #         break
    #     if video_path in readed:
    #         readed[video_path].append(frame)
    #         if index == 63:
    #             print('all frames for',video_path,len(readed[video_path]))
    #     else:
    #         readed[video_path] = []
    #         readed[video_path].append(frame)
    # print('all end')


if __name__ == '__main__':
    video_txt = 'test_multi_video.txt'
    st = time.time()
    test_video(video_txt)
    print(time.time()-st)
