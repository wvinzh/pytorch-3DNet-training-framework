#-*- coding: UTF-8 -*- 
import os
import pims
import test_video
import threading
import logging
import time
import cv2
import asyncio
import random
from utils.opts import parse_opts
from imutils.video import FileVideoStream
from models import resnext
from mytransforms.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import torch.nn as nn
import torch
import imageio
from multiprocessing import Pool
from imutils.video import FPS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#设置只可见gpu1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

NUM_CLIP = 4
NUM_PER_CLIP = 16
INTER_FRAME = 5
frames = []

#start_time = time.time()
#V = pims.Video('/home/zengh/Dataset/AIChallenger/group5/1004255246.mp4')
#total_frame_num = len(V)
#duration = time.time() - start_time
#print('pims video time %.3f s' % duration)  

def initial_model():
  opt = parse_opts()
  model = resnext.resnet101(
              num_classes=opt.n_finetune_classes,
              shortcut_type=opt.resnet_shortcut,
              cardinality=opt.resnext_cardinality,
              sample_size=opt.sample_size,
              sample_duration=opt.sample_duration)

  model = model.cuda()
  model = nn.DataParallel(model, device_ids=None)
  model_path = './trained_models/best.pth10.tar'
  if not os.path.exists(model_path):
    print("model path is not true!!")
    return

  model.load_state_dict(torch.load('./trained_models/best.pth10.tar')['state_dict'])
  model.eval()
  return model

# rgb_mean = [0.485, 0.456, 0.406]
# rgb_std = [0.229, 0.224, 0.225]
# opt.scales=[1]
# transform_val = Compose([
#     MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
#     ToTensor(),
#     Normalize(rgb_mean, rgb_std),
# ])

#记得去掉每行后面的\n
with open('val_group5.txt','r') as f:
    videos = f.readlines()

#NUM_VIDEO = len(videos)
NUM_VIDEO = 50

for i in range(NUM_VIDEO):
  videos[i] = videos[i].strip()
  if not os.path.exists(videos[i]):
    print("video not exists!")
    break

#NUM_VIDEO = 100
NUM_PROCESS = 1
VIDEO_PER_THREAD = int(NUM_VIDEO / NUM_PROCESS)

def decode_video(path, filename, format_out):
    """ Decodes the selected video file and returns the corresponding tensor
        as a uint8 list of numpy arrays.

        Args:
            path (string) : path to the directory containing the file to be decoded.
            filename (string) : string containing the file name.
            format_out (string) : string containing the file extension of the file to be decoded.

        Returns:
            list of numpy_array
    """

    if filename.endswith(format_out):
        vid = imageio.get_reader(path+filename,'ffmpeg')
        nframes = len(vid)
        list = []
        for i in range(nframes):
            try:
                list.append(vid.get_data(i))
            except:
                e = sys.exc_info()[0]
                print('error {0} at index {1}'.format(e,i))
    else:
        print('There is a non-{} file\n'.format(format_out))

    return(list)

# start_time = time.time()
# V = decode_video('/home/zengh/Dataset/AIChallenger/group5/','1004255246.mp4' ,'mp4')
# duration = time.time() - start_time
#print('decode video time %.3f s' % duration)  

def get_logger():
    logger = logging.getLogger("threading_example")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("threading.log")
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

def predict_clip_thread(sindex,logger):
  logger.debug('start predicting')
  #s_index = sindex
  s_index = random.randint(0, sindex)
  list_arr = []
  
  for j in range(s_index, s_index + INTER_FRAME*NUM_PER_CLIP, INTER_FRAME):
    logger.debug('j: {}'.format(j))
    list_arr.append(V[j])
  
  indexes = test_video.predict(model,list_arr,transform=transform_val)
  logger.debug('result: {}'.format(indexes))

def print_test(n,logger):
  for i in range(n):
    logger.debug('i: {}'.format(i))
    #print(i)

def test_thread():
  start_time = time.time()
  logger = get_logger()
  thread_names = range(1,NUM_CLIP+1)
  threads = []
  total_frame_num = len(V)
  #print("total_frame_num",total_frame_num)
  sindex = total_frame_num - INTER_FRAME * NUM_PER_CLIP + 1
  #sindex = [0,76,152]
  #start_time = time.time()
  for i in range(NUM_CLIP):
    t = threading.Thread(target=predict_clip_thread,name=thread_names[i],args=(sindex,logger))
    #t = threading.Thread(target=print_test,name=thread_names[i],args=(10,logger)) 
    threads.append(t)

  for t in threads:
    t.setDaemon(True)
    t.start()
  
  for t in threads:
    t.join()
  
  duration = time.time() - start_time
  print('test time %.3f s' % duration)  

def predict_clip(_sindex):
  sindex = random.randint(0, _sindex)
  list_arr = []
  #for j in range(sindex, sindex + INTER_FRAME*NUM_PER_CLIP, INTER_FRAME):
  for j in range(64):
    list_arr.append(V[j])
  start_time = time.time()
  indexes = test_video.predict(model,list_arr,transform=transform_val)
  duration = time.time() - start_time
  print('test time %.3f s' % duration)  
  print("result",indexes)

def test():
  start_time = time.time()
  total_frame_num = len(V)
  #print("total_frame_num",total_frame_num)
  s_index = total_frame_num - INTER_FRAME * NUM_PER_CLIP + 1
  #for i in range(NUM_CLIP):
  predict_clip(s_index)
  duration = time.time() - start_time
  #print('test time %.3f s' % duration)  

async def predict_clip_asy(sindex):
  s_index = random.randint(0, sindex)
  list_arr = []
  
  for j in range(s_index, s_index + INTER_FRAME*NUM_PER_CLIP, INTER_FRAME):
    list_arr.append(V[j])

  print("start predict_clip_asy ", s_index )
  
  indexes = test_video.predict(model,list_arr,transform=transform_val)
  await asyncio.sleep(1)
  #indexes = []
  return indexes

async def result_clip_async(s_index):
    result = await predict_clip_asy(s_index)
    print(result)

#取的连续帧list
def result_predict(frames,model):
  start_time = time.time()
  frame_num = len(frames)
  #print("frame num is ",frame_num)
  temp = int(frame_num / (NUM_PER_CLIP * NUM_CLIP))
  predict_frame = []
  for i in range(NUM_PER_CLIP * NUM_CLIP):
      predict_frame.append(frames[i * temp])

  result = test_video.predict(model,predict_frame)
  #print("result:",result)
  duration = time.time() - start_time
  #print('test time %.3f s' % duration)  
  return result

#先读图片,再预测,图片写入队列作为一个线程
def videostream_predict(video_index,process_num,model):
  try:
    # start the FPS timer
    print('Run task %s (%s)...' % (process_num, os.getpid()))
    st2 = time.time()
    clip = 0
    video_num = 0
    total_video_num = len(video_index)
    frames = []
    video = videos[video_index[video_num]]
    if not os.path.exists(video):
        print("video not exists!")
        return

    fvs = FileVideoStream(video,resize=112).start()

    while video_num < (total_video_num - 1):
      #得到的是图片tensor
      st4 = time.time()
      frame = fvs.read()
      #print("Q size is",fvs.more())
      #print("read time is ",time.time() - st4)
      #clip += 1 
      #耗时少
      frames.append(frame)
      #print("Process ",process_num," clip ",clip," is more ",fvs.more())
      
      #if clip == 200 or frame.dim() == 1:
      #遇到终止帧
      if frame.dim() == 1:
        if video_num == 0:
          st1 = time.time()
        #print("the part is ",time.time() - st1)
        res_frames = frames.copy()  #深拷贝
        
        fvs.stop()
        #clip = 0
        video_num = video_num + 1
        video = videos[video_index[video_num]]
        if not os.path.exists(video):
          print("video not exists!")
          break
        fvs = FileVideoStream(video,resize=112).start()

        #去掉终止帧
        res_frames.pop(-1)

        frames = []	
        #print("start predict**",process_num)
        #读取下一个视频部分的时间,可以读超过128帧的视频帧(读取帧率和视频有关?),所以可以调节队列maxsize的大小
        st = time.time()
        indexes = result_predict(res_frames,model)
        #print("predict time is ",time.time() - st)
        
        st1 = time.time()
        #print("finish predict**",process_num)

      
    fvs.stop()
    print('Task %s runs %0.2f seconds.' % (process_num, (time.time() - st2)))
  
  except Exception as ex:
    msg = "error:%s"%ex
    print(ex)

#图片的写入队列和从队列中取作为两个线程同时进行
def videostream_predict_1(video_index,process_num,model):
  try:
    # start the FPS timer
    print('Run task %s (%s)...' % (process_num, os.getpid()))
    
    video_num = 0
    total_video_num = len(video_index)
    video = videos[video_index[video_num]]
    if not os.path.exists(video):
        print("video not exists!")
        return
    #st2 = time.time()
    fvs = FileVideoStream(video,resize=112).start_produce_consume()
    #print("read time is ",time.time() - st2)
    #print(fvs.if_read_end())

    st1 = time.time()
    while video_num < (total_video_num - 1):   
      if fvs.if_read_end() == True:
        print("read time is ",time.time() - st1)
        frames = fvs.get_frames()
        print("video len is ",len(frames))
         
        video_num = video_num + 1
        video = videos[video_index[video_num]]
        if not os.path.exists(video):
          print("video not exists!")
          break

        fvs = FileVideoStream(video,resize=112).start_produce_consume()
        indexes = result_predict(frames,model)
        st1 = time.time()
        #print("video ",video_num," indexes",indexes)
      
    print('Task %s runs %0.2f seconds.' % (process_num, (time.time() - st2)))
  
  except Exception as ex:
    msg = "error:%s"%ex
    print(ex)

def multiprocess_predict():
  st = time.time()
  print('Parent process %s.' % os.getpid())
  p = Pool(NUM_PROCESS)
  model = initial_model()
  print("model has been restored!",time.time() - st)
  for i in range(NUM_PROCESS):
    index = range(i*VIDEO_PER_THREAD, (i+1)*VIDEO_PER_THREAD)
    if i == (NUM_PROCESS - 1):
      index = range(i*VIDEO_PER_THREAD,NUM_VIDEO)

    p.apply_async(videostream_predict_1, args=(index,i,model, ))

  print('Waiting for all subprocesses done...')
  p.close()
  p.join()
  print('All subprocesses done. %0.2f seconds.' % (time.time() - st))
  
if __name__=='__main__':
    # start_time = time.time()
    # s_index = total_frame_num - INTER_FRAME * NUM_PER_CLIP + 1

    # loop = asyncio.get_event_loop()
    
    # loop.run_until_complete(asyncio.gather(result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index),
    #                                       result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index),result_clip_async(s_index)))
    
    # #loop.run_until_complete(asyncio.gather(result_clip_async(s_index)))
    # loop.close()
    # duration = time.time() - start_time
    # print('test time %.3f s' % duration)  
    multiprocess_predict()
    #test()

  

    