import threading
import time
import pims,cv2
import multiprocessing
import queue
from PIL import Image

def get_frames(v_f, num):
    print(multiprocessing.current_process().name,'get %d frame of video'%num,v_f[num].shape)
    # print(num,)
    return v_f[num]

#具体做啥事,写在函数中
def run(number):
    print(threading.currentThread().getName() + '\n')
    print(number)

def test_cv(frame_queue):
    myFrameNumber = 50
    st=time.time()
    cap = cv2.VideoCapture("/home/zengh/Dataset/AIChallenger/group5/777770896.mp4")
    print(time.time()-st)
    # get total number of frames
    st=time.time()
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('get total',time.time()-st)
    # check for valid frame number
    # if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    #     # set frame position
    #     tmp_time = time.time()
    #     # cap.set(cv2.CAP_PROP_POS_FRAMES,100)
    #     print('set frame',time.time()-tmp_time)
    total = 0
    tmp_time = time.time()
    while True:
        # tmp_time = time.time()
        ret, frame = cap.read()
        # print('read',time.time()-tmp_time)
        if not ret:
            break
        total+=1
        # frame_queue.put((total,frame))
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break
        # cv2.imshow("Video", frame)
    print('read',time.time()-tmp_time)
    cv2.destroyAllWindows()

def main():
    num_of_clip = 4
    start = time.time()
    video = pims.Video('/home/zengh/Dataset/AIChallenger/group5/777770896.mp4')
    # print(video)
    print('read',time.time()-start)
   
    clip_step = int(len(video)/num_of_clip)
    for i in range(num_of_clip):
        if i == 0:
            indes = i
        else:
            indes = i+100
        # my_thread = threading.Thread(target=get_frames, args=(video,indes,))
        # my_thread.start()
        mp = multiprocessing.Process(target=get_frames, args=(video,indes,))
        mp.start()
        mp.join()

def read_one_frame():
    cap = cv2.VideoCapture("/home/zengh/Dataset/AIChallenger/group5/777770896.mp4")
    _,frame = cap.read()

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(frame)
    pil_im_rgb = Image.fromarray(frame_rgb)
    pil_im.show()
    pil_im_rgb.show()
    # print(frame)
    # cv2.imshow('f',frame)
    # cv2.waitKey(0)

if __name__ == '__main__':
    start = time.time()
    # main()
    read_one_frame()
    # print(fq.get(),fq.get())
    print(time.time()-start)
