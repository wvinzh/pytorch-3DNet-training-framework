import torch
import random,os
import glob


def get_video_names_and_annotations(annotation_path):
    video_names = []
    annotations = []
    with open(annotation_path, 'r') as f:
        line = f.readline()
        while line:
            # print(line)
            video_names.append(line.strip().split(' ', 1)[0])
            label_list = (line.strip().split(' ', 1)[1]).split(',')
            annotations.append([int(k) for k in label_list])
            line = f.readline()
    return video_names, annotations

# get_video_names_and_annotations('/home/zengh/Dataset/AiChallengerFrames/300_val.txt')


train_num = [9565, 9253, 624, 760, 890, 9226, 482, 2168, 11254, 1720,
             17420, 500, 313, 593, 653, 240, 729, 8295, 9763, 5607,
             823, 285, 409, 455, 605, 452, 395, 373, 8507, 291,
             322, 484, 332, 454, 1138, 5831, 5514, 2981, 332, 506,
             488, 5605, 2067, 293, 98, 508, 922, 882, 822, 319,
             248, 282, 426, 466, 449, 368, 456, 384, 518, 432, 494, 345, 790]

val_num = [2116, 2066, 243, 295, 245, 2061, 134, 672, 2477, 475,
           3691, 132, 91, 156, 187, 106, 189, 1883, 2120, 1426,
           237, 64, 134, 140, 202, 143, 129, 129, 1897, 99,
           87, 150, 104, 141, 295, 1623, 1271, 833, 129, 144,
           134, 1445, 672, 67, 39, 136, 255, 249, 231, 80,
           57, 87, 124, 122, 127, 111, 129, 113, 140, 123, 133, 99, 268]


def random_list(n):
    for i in range(10):
        sample = random.sample(range(i, n+i), 16)
        print(sorted(sample))

def create_trainval():
    numclass = 63
    num_of_each = 600
    base_n_of_sample = 5
    input_depth = 16
    trainval_txt = open('trainval.txt','a')

    for i in range(numclass):
        txt = '/home/zengh/Dataset/AiChallengerFrames/train_600/train_%d_%d.txt' % (num_of_each,i)
        lines = open(txt,'r').readlines()
        total = len(lines)
        if total >= num_of_each:
            n_of_sample = base_n_of_sample
        else:
            n_of_sample = int(base_n_of_sample * float(num_of_each/total))+1
        for line in lines:
            video_name = line.strip().split()[0]
            video_label = line.strip().split()[1]
            frame_path = os.path.join('/home/zengh/Dataset/AIChallenger/train',video_name.split('.')[0])
            jpg_path = os.path.join(frame_path,'*.jpg')
            # print(jpg_path)
            jpg_list = glob.glob(jpg_path)
            jpg_total  = len(jpg_list)-2-1
            if jpg_total < input_depth:
                continue
            
            sample_step = (jpg_total - input_depth) // n_of_sample
            randnum = jpg_total - sample_step*n_of_sample -input_depth
            if sample_step <= 0:
                for j in range(5):
                    slices = sorted(random.sample(range(1,jpg_total+1),input_depth))
                    trainval_txt.write('%s %s %s\n'%(video_name,video_label,'#'.join(str(e) for e in slices)))
                    tmp_jpg = os.path.join(frame_path,'%05d.jpg'%slices[input_depth-1])
                    # print(tmp_jpg)
                    if not os.path.exists(tmp_jpg):
                        print(slices,line)
                        continue
            else:
                for j in range(n_of_sample):
                    slices = sorted(random.sample(range(j*sample_step+1,input_depth+(j+1)*sample_step+randnum),input_depth))
                    tmp_jpg = os.path.join(frame_path,'%05d.jpg'%slices[input_depth-1])
                    # print(tmp_jpg)
                    if not os.path.exists(tmp_jpg):
                        print(slices,line)
                        continue
                    trainval_txt.write('%s %s %s\n'%(video_name,video_label,'#'.join(str(e) for e in slices)))

def seperate_trainval():

    lines = open('trainval.txt','r').readlines()
    random.shuffle(lines)
    total = len(lines)
    train_num = int(total*0.8)
    open('600_train.txt','a').writelines(lines[0:train_num])
    open('600_val.txt','a').writelines(lines[train_num::])




def chech_not_in():
    o_txt = '/home/zengh/Dataset/AiChallengerFrames/short_video_validationset_annotations.txt'
    n_txt = '/home/zengh/Dataset/AiChallengerFrames/new_test.txt'
    o_lines = [x.strip().split(',',1)[0] for x in open(o_txt,'r')]
    n_lines = [y.strip().split()[0].split('/')[1] for y in open(n_txt,'r')]
    for i,l in enumerate(o_lines):
        if l not in n_lines:
            print(l)

# chech_not_in()

def one_label():
    numclass = 63
    num_of_each = 600
    one_l = [0]*numclass
    for i in range(numclass):
        txt = '/home/zengh/Dataset/AiChallengerFrames/train/train_label_%d.txt' % i
        with open(txt,'r') as f:
            line = f.readline()
            while line:
                if len(line.strip().split()[1].split(',')) <= 1:
                    one_l[i] += 1
                line = f.readline()
        # print(i,one_l[i])
    for i in range(numclass):
        txt = '/home/zengh/Dataset/AiChallengerFrames/train/train_label_%d.txt' % i
        txt_out = '/home/zengh/Dataset/AiChallengerFrames/train_600/train_%d_%d.txt' % (num_of_each,i)
        of = open(txt_out,'a')
        with open(txt,'r') as f:
            lines = f.readlines()
            
            if one_l[i] <= num_of_each:
                of.writelines(lines)
            else:
                a = num_of_each
                random.shuffle(lines)
                for line in lines:
                    if len(line.strip().split()[1].split(',')) <= 1:
                        of.write(line)
                        a-=1
                        if a <=0:
                            break
                    




# one_label()
# create_trainval()  
seperate_trainval()  
