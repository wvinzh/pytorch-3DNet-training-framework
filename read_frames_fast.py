#-*- coding: UTF-8 -*- 
# USAGE
# python read_frames_fast.py --video videos/jurassic_park_intro.mp4

# import the necessary packages
# from imutils.video import FileVideoStream
# from imutils.video import FPS
# import numpy as np
# import argparse
# import imutils
# import time
# import cv2
# import time
# import pylab
# import pims
# #pylab.switch_backend('agg')



# start_time = time.time()
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,
# 	help="path to input video file")

# args = vars(ap.parse_args())

# # start the file video stream thread and allow the buffer to
# # start to fill
# print("[INFO] starting video file thread...")
# st = time.time()
# fvs = FileVideoStream(args["video"]).start()
# print('start time',time.time()-st)
# #time.sleep(0.5)

# NUM_CLIP = 1
# NUM_PER_CLIP = 16
# INTER_FRAME = 5

# # start the FPS timer
# fps = FPS().start()
# clip = 0
# total = 0
# frames = []
# # loop over frames from the video file stream
# #while fvs.more():
# while total < 3:
# #while True:
# 	# grab the frame from the threaded video file stream, resize
# 	# it, and convert it to grayscale (while still retaining 3
# 	# channels)
# 	frame = fvs.read()
# 	clip = clip + 1
# 	#print("clip",clip)
# 	#print(total)
# 	frames.append(frame)

# 	if clip == 200:
# 		#print(clip)
# 		fvs.stop()
# 		clip = 0
# 		fvs = FileVideoStream('/home/zengh/Dataset/AIChallenger/group5/994513477.mp4').start()	
# 		#temp = frame[0:300]
# 		#temp = frame[total*16:total*16+INTER_FRAME*5:INTER_FRAME]
# 		#temp = frame[total*80:(total+1)*80
# 		total = total + 1
# 		#print("start predict***************************")
# 		#print("frame num",len(frames))
# 		time.sleep(0.5) #读取下一个视频部分的时间,可以读超过128帧的视频帧(读取帧率和视频有关?),所以可以调节队列maxsize的大小
# 		#print("finish predict***************************")
# 		frames = []
# 		#break
# 	fps.update()

# # stop the timer and display FPS information
# fps.stop()
# print('end,total %d '% total,time.time()-st)
# print("[INFO] elasped time: {:.8f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.8f}".format(fps.fps()))

# # do a bit of cleanup
# #cv2.destroyAllWindows()
# fvs.stop()

# from multiprocessing import Pool
# import os, time, random

# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))

# if __name__=='__main__':
#     st = time.time()
#     print('Parent process %s.' % os.getpid())
#     p = Pool(20)
#     for i in range(20):
#         p.apply_async(long_time_task, args=(i,))
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')
#     print('TAll subprocesses done. %0.2f seconds.' % (time.time() - st))

import multiprocessing
import time

def func(msg):
    for i in range(3):
        print(msg)
        time.sleep(1)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    for i in range(10):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")