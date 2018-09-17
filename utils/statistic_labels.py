
import os
import zipfile
import glob
import re
def statistics_label(label_root,file_name):
    filepath = os.path.join(label_root,file_name)
    result = [0]*63
    total = 0
    one_label = 0
    if not os.path.exists(filepath):
        print('not exist')
    with open(filepath,'r') as f:
        line = f.readline()
        while line:
            labels = line.strip().split(',',1)[1]
            labels = labels.split(',')
            n_label = len(labels)
            total += 1
            if n_label ==1:
                one_label += 1
            # for i in labels:
            #     label = int(i)
            result[int(labels[0])] += 1
            line = f.readline()
    print(total,one_label,result)
    return result

def extract_names(label_root,file_name):
    filepath = os.path.join(label_root,file_name)
    result = [0]*63
    names = {}
    if not os.path.exists(filepath):
        print('not exist')
    with open(filepath,'r') as f:
        line = f.readline()
        while line:
            name = line.strip().split(',',1)[0]
            _label = line.strip().split(',',1)[1]
            video_info = (name,_label)
            labels = _label.split(',')
            for i in labels:
                label = int(i)
                result[label] += 1
                if i not in names:
                    names[i] = []
                else:
                    names[i].append(video_info)
            # print(names.keys())
            line = f.readline()
    # print(names['44'])
    for k,v in names.items():
        label_file_name = 'test_label_%s.txt' % k
        label_file_path = os.path.join('/home/zengh/Dataset/AiChallengerFrames/test',label_file_name)
        if os.path.exists(label_file_path):
            os.remove(label_file_path)
        print(k,len(v))
        with open(label_file_path,'w') as f:
            label_names = ['%s %s\n' % (nm[0],nm[1]) for nm in v]
            f.writelines(label_names)
    return names,result

def scan_zip(root_path,label_txt,out_root):
    zipfile_list = glob.glob(pathname=os.path.join(root_path,'*.zip'))
    zipfile_list.sort(reverse=True)
    print(zipfile_list)
    origin_file = open(label_txt,'r')
    origin_lines = origin_file.readlines()
    # target_lines = ['0']*len(origin_lines)
    target_file = open('new_test.txt','w')
    for one_zip in zipfile_list:
        print('scan zip file: %s'% one_zip)
        zf = zipfile.ZipFile(one_zip)
        for mp4_file in zf.namelist():
            if not mp4_file.endswith('mp4'):
                continue
            origin_name = mp4_file.split('/')[1]
            target_name = mp4_file
            for origin_line in origin_lines[:]:
                if len(re.findall(origin_name,origin_line))>0:
                    target_file.write(origin_line.replace(origin_name,target_name))
                    origin_lines.remove(origin_line)
                    break
    # target_file.writelines(origin_lines)
    origin_file.flush()
    origin_file.close()
    target_file.flush()
    target_file.close()



extract_names('/home/zengh/AiChallenger/Video-Classification-Pytorch-FrameWork/','new_test.txt')
# scan_zip('/home/zengh/Dataset/AIChallenger/test','/home/zengh/Dataset/AIChallenger/short_video_validationset_annotations.txt.0829','')